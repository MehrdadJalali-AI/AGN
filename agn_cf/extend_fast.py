"""Local target-logit computation for one inserted node. Same equations as the full stack."""
import torch
import torch.nn.functional as F


class Cache:
    def __init__(self, model, x, edge_index):
        self.model = model
        self.kind = model.__class__.__name__
        n = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        if torch.any(src == dst):
            raise RuntimeError("self-loops are already present; the local degree formula does not apply")
        order = torch.argsort(dst)
        self.neigh = src[order]
        counts = torch.bincount(dst, minlength=n)
        self.ptr = torch.zeros(n + 1, dtype=torch.long)
        self.ptr[1:] = counts.cumsum(0)
        self.deg = counts.float()
        self.x = x
        self.n = n
        with torch.no_grad():
            if self.kind == "GCN":
                self._cache_gcn()
            elif self.kind == "GAT":
                self._cache_gat()
            elif self.kind == "SAGE":
                self._cache_sage()
            else:
                raise RuntimeError(f"no local logits for {self.kind}")

    def nbrs_of(self, i):
        return self.neigh[self.ptr[i]:self.ptr[i + 1]]

    def _cache_gcn(self):
        conv = self.model.conv1
        z = conv.lin(self.x)
        inv = (self.deg + 1).pow(-0.5)
        src_term = inv[self.neigh].unsqueeze(1) * z[self.neigh]
        s = torch.zeros_like(z)
        s.index_add_(0, self._dst_index(), src_term)
        s = s + inv.unsqueeze(1) * z
        out = inv.unsqueeze(1) * s + conv.bias
        self.z1, self.inv, self.s1 = z, inv, s
        self.h1 = out.relu()
        conv2 = self.model.conv2
        self.z2 = conv2.lin(self.h1)

    def _dst_index(self):
        # neigh is sorted by destination, so the destination of slice ptr[i]:ptr[i+1] is i
        return torch.repeat_interleave(torch.arange(self.n), self.ptr[1:] - self.ptr[:-1])

    def _cache_sage(self):
        conv = self.model.conv1
        z = F.linear(self.x, conv.lin_l.weight)
        s = torch.zeros_like(z)
        s.index_add_(0, self._dst_index(), z[self.neigh])
        count = self.deg.clamp(min=1).unsqueeze(1)
        mean = torch.where(self.deg.unsqueeze(1) > 0, s / self.deg.unsqueeze(1), torch.zeros_like(s))
        h = mean + conv.lin_l.bias + F.linear(self.x, conv.lin_r.weight)
        self.sz1 = s
        self.h1 = h.relu()
        conv2 = self.model.conv2
        z2 = F.linear(self.h1, conv2.lin_l.weight)
        s2 = torch.zeros_like(z2)
        s2.index_add_(0, self._dst_index(), z2[self.neigh])
        self.sz2 = s2

    def _cache_gat(self):
        conv = self.model.conv1
        y = F.linear(self.x, conv.lin.weight, conv.lin.bias).view(-1, conv.heads, conv.out_channels)
        att_s = conv.att_src.view(conv.heads, conv.out_channels)
        att_d = conv.att_dst.view(conv.heads, conv.out_channels)
        self.gx, self.ga_s, self.ga_d = y, (y * att_s).sum(-1), (y * att_d).sum(-1)
        with torch.no_grad():
            full = self.model.conv1(self.x, self._edge_index())
        self.h1 = F.elu(full).detach()
        self.y2, self.as2, self.ad2 = self._gat_pack_many(self.model.conv2, self.h1)

    def _edge_index(self):
        src = self.neigh
        dst = torch.repeat_interleave(torch.arange(self.n), self.ptr[1:] - self.ptr[:-1])
        return torch.stack([src, dst], 0)

    def logits(self, feat, nbrs):
        nbrs = [int(v) for v in nbrs]
        if self.kind == "GCN":
            return self._gcn_logits(feat, nbrs)
        if self.kind == "SAGE":
            return self._sage_logits(feat, nbrs)
        return self._gat_logits(feat, nbrs)

    def _gcn_logits(self, feat, nbrs):
        target = nbrs[0]
        conv1, conv2 = self.model.conv1, self.model.conv2
        z_new = conv1.lin(feat)
        inv_new = feat.new_tensor(len(nbrs) + 1).pow(-0.5)
        inv_s = (self.deg[nbrs] + 2).pow(-0.5)
        R = torch.unique(torch.cat([self.nbrs_of(target), torch.tensor([target])]))
        pos = torch.full((self.n,), -1, dtype=torch.long)
        pos[R] = torch.arange(R.numel())
        local = self.s1[R].clone()
        member = torch.zeros(self.n, dtype=torch.bool)
        member[nbrs] = True
        for j, node in enumerate(nbrs):
            delta = (inv_s[j] - self.inv[node]) * self.z1[node]
            nbr = self.nbrs_of(node)
            hit = nbr[pos[nbr] >= 0]
            if hit.numel():
                local.index_add_(0, pos[hit], delta.expand(hit.numel(), -1))
            pj = int(pos[node])
            if pj >= 0:
                local[pj] = local[pj] + delta + inv_new * z_new
        inv_used = self.inv[R].clone()
        in_r = pos[nbrs] >= 0
        inv_used[pos[nbrs][in_r]] = inv_s[in_r]
        out1 = inv_used.unsqueeze(1) * local + conv1.bias
        h = out1.relu()
        # new node
        s_new = (inv_s.unsqueeze(1) * self.z1[nbrs]).sum(0) + inv_new * z_new
        h_new = (inv_new * s_new + conv1.bias).relu()
        # layer 2 at the target, using updated h for R and the new node
        h_map = self.h1.clone()
        h_map[R] = h
        z = conv2.lin(h_map)
        z_new2 = conv2.lin(h_new)
        inv_t = (self.deg[target] + 2).pow(-0.5)
        nbr_t = self.nbrs_of(target)
        inv_n = self.inv[nbr_t].clone()
        inv_n[member[nbr_t]] = (self.deg[nbr_t][member[nbr_t]] + 2).pow(-0.5)
        s2 = inv_t * z[target] + (inv_n.unsqueeze(1) * z[nbr_t]).sum(0) + inv_new * z_new2
        return inv_t * s2 + conv2.bias

    def _sage_h(self, conv, sum_z, count, root):
        count = count.to(dtype=sum_z.dtype)
        if count.ndim > 0:
            count = count.unsqueeze(-1)
        mean = torch.where(count > 0, sum_z / count.clamp(min=1), torch.zeros_like(sum_z))
        return mean + conv.lin_l.bias + F.linear(root, conv.lin_r.weight)

    def _sage_logits(self, feat, nbrs):
        target = nbrs[0]
        c1, c2 = self.model.conv1, self.model.conv2
        z_new = F.linear(feat, c1.lin_l.weight)
        z_nbrs = F.linear(self.x[nbrs], c1.lin_l.weight).sum(0)
        h_new = self._sage_h(c1, z_nbrs, feat.new_tensor(float(len(nbrs))), feat)
        # existing nodes in nbrs gain feat as one neighbor
        base = self.sz1[nbrs] + z_new
        counts = self.deg[nbrs] + 1
        h_s = self._sage_h(c1, base, counts, self.x[nbrs])
        h_s = h_s.relu()
        h_new = h_new.relu()
        # layer 2
        z2_s = F.linear(h_s, c2.lin_l.weight)
        z2_new = F.linear(h_new, c2.lin_l.weight)
        # map node -> row in nbrs
        loc = torch.full((self.n,), -1, dtype=torch.long)
        loc[nbrs] = torch.arange(len(nbrs))
        nbr_t = self.nbrs_of(target)
        use = loc[nbr_t]
        z_nbr = self.sz2.new_zeros(nbr_t.numel(), z2_s.size(1)) if False else F.linear(self.h1[nbr_t], c2.lin_l.weight)
        changed = use >= 0
        if changed.any():
            z_nbr = z_nbr.clone()
            z_nbr[changed] = z2_s[use[changed]]
        # cached neighbor sum uses old z2; rebuild the target sum from pieces
        # original sum is sz2[target]; replace changed neighbors and add the new node
        old_z = F.linear(self.h1[nbr_t], c2.lin_l.weight)
        sum_z = old_z.sum(0)
        if changed.any():
            sum_z = sum_z - old_z[changed].sum(0) + z_nbr[changed].sum(0)
        sum_z = sum_z + z2_new
        count = self.deg[target] + 1
        h_self = h_s[loc[target]]
        return self._sage_h(c2, sum_z, count, h_self)

    def _gat_pack_many(self, conv, feat):
        y = F.linear(feat, conv.lin.weight, conv.lin.bias).view(-1, conv.heads, conv.out_channels)
        att_s = conv.att_src.view(conv.heads, conv.out_channels)
        att_d = conv.att_dst.view(conv.heads, conv.out_channels)
        return y, (y * att_s).sum(-1), (y * att_d).sum(-1)

    def _gat_pack(self, conv, feat):
        y, a_s, a_d = self._gat_pack_many(conv, feat.unsqueeze(0))
        return y[0], a_s[0], a_d[0]

    def _gat_node(self, conv, y_rows, a_rows, a_dst):
        # y_rows (m, heads, C), a_rows (m, heads), a_dst (heads,)
        scores = F.leaky_relu(a_rows + a_dst, conv.negative_slope)
        alpha = torch.softmax(scores, dim=0)
        out = (alpha.unsqueeze(-1) * y_rows).sum(0)
        if conv.concat:
            out = out.reshape(-1)
        else:
            out = out.mean(0)
        return out + conv.bias

    def _gat_logits(self, feat, nbrs):
        target = nbrs[0]
        c1, c2 = self.model.conv1, self.model.conv2
        y_new, as_new, ad_new = self._gat_pack(c1, feat)
        h_rows = []
        for node in nbrs:
            nbr = self.nbrs_of(node)
            y_rows = torch.cat([self.gx[node].unsqueeze(0), self.gx[nbr], y_new.unsqueeze(0)], 0)
            a_rows = torch.cat([self.ga_s[node].unsqueeze(0), self.ga_s[nbr], as_new.unsqueeze(0)], 0)
            h_rows.append(F.elu(self._gat_node(c1, y_rows, a_rows, self.ga_d[node])))
        h_s = torch.stack(h_rows, 0)
        # new node attends over nbrs and itself
        y_nb = self.gx[nbrs]
        a_nb = self.ga_s[nbrs]
        y_rows = torch.cat([y_nb, y_new.unsqueeze(0)], 0)
        a_rows = torch.cat([a_nb, as_new.unsqueeze(0)], 0)
        h_new = F.elu(self._gat_node(c1, y_rows, a_rows, ad_new))
        y_s, as_s, ad_s = self._gat_pack_many(c2, h_s)
        y_new, as_new, _ = self._gat_pack(c2, h_new)
        loc = torch.full((self.n,), -1, dtype=torch.long)
        loc[nbrs] = torch.arange(len(nbrs))
        nbr_t = self.nbrs_of(target)
        use = loc[nbr_t]
        y_nbr = self.y2[nbr_t].clone()
        a_nbr = self.as2[nbr_t].clone()
        changed = use >= 0
        if changed.any():
            y_nbr[changed] = y_s[use[changed]]
            a_nbr[changed] = as_s[use[changed]]
        y_rows = torch.cat([y_s[loc[target]].unsqueeze(0), y_nbr, y_new.unsqueeze(0)], 0)
        a_rows = torch.cat([as_s[loc[target]].unsqueeze(0), a_nbr, as_new.unsqueeze(0)], 0)
        return self._gat_node(c2, y_rows, a_rows, ad_s[loc[target]])


def attach(model, x, edge_index):
    model._fast = Cache(model, x, edge_index)
    return model._fast


def forward_logits(model, x, edge_index, feat, nbrs):
    target = int(nbrs[0])
    scores = model._fast.logits(feat, nbrs)
    out = scores.new_zeros(x.size(0) + 1, scores.numel())
    index = torch.tensor([target])
    return out.index_copy(0, index, scores.unsqueeze(0))

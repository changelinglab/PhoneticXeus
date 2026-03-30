# Implementation of Label Prior CTC (https://arxiv.org/abs/2406.02560v3)
# forward_core from https://github.com/huangruizhe/audio/blob/aligner_label_priors/examples/asr/librispeech_alignment/loss.py

import logging
import k2
import torch


class LabelPriorCTC(torch.nn.Module):
    def __init__(self, prior_scaling_factor):
        super().__init__()
        self.prior_scaling_factor = prior_scaling_factor
        self.hs_priors = None
        self.hs_priors_sum = None

    def forward(self, hs_pad, ys_pad, hlens, ylens):
        indices = torch.argsort(hlens, descending=True)
        ys, min_hlens = self.find_minimum_hlens(ys_pad[indices], ylens[indices])
        valid_sample_indices = (min_hlens <= hlens[indices]).nonzero(as_tuple=True)[0]

        if len(valid_sample_indices) < 1:
            logging.warning(
                "All examples are invalid for Label Prior CTC. Skip this batch"
            )
            return torch.Tensor([0.0]).to(hs_pad.device)

        indices = indices[valid_sample_indices]
        hs_pad, hlens, ylens = hs_pad[indices], hlens[indices], ylens[indices]
        ys = [ys[i.item()] for i in valid_sample_indices]

        # Core implementation
        loss = self.forward_core(hs_pad, ys, hlens, ylens)

        return loss, len(indices)

    def forward_core(self, hs_pad, ys, hlens, ylens):
        # Find the shape
        (B, T, _), U = hs_pad.size(), max(ylens)

        # Build CTC graphs
        supervision = torch.stack(
            [torch.arange(B), torch.zeros(B), hlens.cpu()], dim=1
        ).int()
        ctc_graphs = k2.ctc_graph(ys).to(hs_pad.device)

        # Accumulate label priors
        hs_flattened = []
        for lp, le in zip(hs_pad, hlens):
            hs_flattened.append(lp[: int(le.item())])
        hs_flattened = torch.cat(hs_flattened, 0)

        hs_priors_sum = torch.logsumexp(hs_flattened, dim=0, keepdim=True)
        hs_priors_sum = hs_priors_sum.detach()
        if self.hs_priors_sum is None:
            self.hs_priors_sum = hs_priors_sum
        else:
            _temp = torch.stack([self.hs_priors_sum, hs_priors_sum], dim=-1)
            self.hs_priors_sum = torch.logsumexp(_temp, dim=-1)

        # Apply label priors and build DenseFsaVec
        if self.hs_priors is not None and self.prior_scaling_factor > 0:
            hs_pad = hs_pad - self.hs_priors * self.prior_scaling_factor
        dense_fsa_vec = k2.DenseFsaVec(hs_pad, supervision)

        # Compute CTC loss
        loss = k2.ctc_loss(
            decoding_graph=ctc_graphs,
            dense_fsa_vec=dense_fsa_vec,
            output_beam=1e20,
        )
        print(loss.shape, loss)
        return loss

    def find_minimum_hlens(self, ys_pad, ylens):
        device = ys_pad.device
        ys_pad, ylens = ys_pad.cpu().tolist(), ylens.cpu().tolist()
        ys, min_hlens = [], []

        for y_pad, ylen in zip(ys_pad, ylens):
            y, min_hlen, prev = [], 0, None
            for i in range(ylen):
                y.append(y_pad[i])
                min_hlen += 1
                if y_pad[i] == prev:
                    min_hlen += 1
                prev = y_pad[i]
            ys.append(y)
            min_hlens.append(min_hlen)

        min_hlens = torch.Tensor(min_hlens).long().to(device)
        return ys, min_hlens

import scipy
import torch

from watermark_benchmark.watermark.templates.generator import Watermark
from watermark_benchmark.watermark.templates.verifier import (
    EmpiricalVerifier,
    Verifier,
)


class BootstrapGeneration(Watermark):
    """
    A watermarking scheme that adds a delta value to the logits of certain tokens in the input sequence. See Kirchenbauer et al. (2023) for more details.

    Args:
        rng (RandomNumberGenerator): A random number generator object.
        verifier (Verifier): A verifier object.
        tokenizer (Tokenizer): A tokenizer object.
        temp (float): Temperature parameter for softmax function.
        delta (float): Value to add to logits of selected tokens.
        gamma (float): Proportion of tokens to select for watermarking.

    Attributes:
        delta (float): Value to add to logits of selected tokens.
        gamma (float): Proportion of tokens to select for watermarking.
        temp (float): Temperature parameter for softmax function.
    """

    def __init__(self, rng, verifier, tokenizer, temp, delta, gamma, K, proposal_temp):
        super().__init__(rng, verifier, tokenizer, temp)
        self.delta = delta
        self.gamma = gamma
        self.temp = temp
        self.K = K
        self.proposal_temp = proposal_temp

    def process(self, logits, previous_tokens, ids):
        """
        Applies the watermarking scheme to the input logits.

        Args:
            logits (torch.Tensor): The input logits.
            previous_tokens (torch.Tensor): The previous tokens in the sequence.
            ids (torch.Tensor): The IDs of the previous tokens.

        Returns:
            torch.Tensor: The logits with the watermark applied.
        """

        # Truncate unused logits
        logits = logits[:, : self.rng.vocab_size]

        N, _ = logits.shape

        # Do not watermark the first K tokens
        if len(previous_tokens[0]) < self.K:
            # print(len(previous_tokens[0]))
            return logits
        previous_K_tokens = [tokens[-self.K:] for tokens in previous_tokens]

        # self.rng.token_compatibility()
        
        # Get crypto seed
        crypto_seed = self.rng.rand_index(self.rng.get_seed(previous_K_tokens, ids), 0)
        # print(crypto_seed, ids)
        
        # Adjust logits using self.temp (temperature)
        logits = logits / self.temp

        # Compute the probability of the original model
        p = torch.nn.functional.softmax(logits, dim=-1)

        # Get seed for the smaller model and proposal logits
        seed = int(crypto_seed[0].item() * 1145141919810) 
        proposal_token, q = self.rng.proposal(previous_K_tokens, seed, temperature=self.proposal_temp)
        q = q.to(p.device)

        # Initialize q_new with zeros
        q_new = torch.zeros_like(q)

        # Setup the acceptance probability
        for i in range(N):
            proposal_index = proposal_token[i]
            if p[i, proposal_index] > q[i, proposal_index]:
                q_new[i, proposal_index] = 1
            else:
                q_new[i, proposal_index] = p[i, proposal_index] / q[i, proposal_index]
                for j in range(q.size(1)):
                    if j != proposal_index:
                        q_new[i, j] = (1 - p[i, proposal_index] / q[i, proposal_index]) * \
                                       (p[i, j] - q[i, j]).clamp(min=0) / \
                                       (p[i] - q[i]).clamp(min=0).sum()
            # print(q_new[i].sum())

        # Convert q_new back to logits
        q_new_logits = torch.log(q_new + 1e-12) * self.temp

        return q_new_logits


class BootstrapVerifier(Verifier):
    """
    A verifier that checks for distribution shift in a sequence of tokens.

    Args:
        rng (RandomNumberGenerator): A random number generator.
        pvalue (float): The p-value threshold for the binomial test.
        tokenizer (Tokenizer): A tokenizer for the sequence of tokens.
        gamma (float): The proportion of tokens that are allowed to be different.

    Attributes:
        gamma (float): The proportion of tokens that are allowed to be different.
    """

    def __init__(self, rng, pvalue, tokenizer, gamma, K, proposal_temp):
        super().__init__(rng, pvalue, tokenizer)
        self.gamma = gamma
        self.K = K
        self.proposal_temp = proposal_temp

    def verify(self, tokens, index=0, exact=False, meta=None):
        tokens = tokens.squeeze()
        cumul = []
        cumul_probs = []
        p_values = []
        seen = set()
        decoded_texts = []

        # print(self.rng.state)

        if not tokens.nelement() or not len(tokens.shape):
            return [(False, 0.5, 0.5, 0, 0)]

        for i in range(self.K, len(tokens)):
            prev_values = tokens[:i].tolist()

            if i < self.K:
                continue
            else:
                prev_K_tokens = tokens[i-self.K:i].tolist()

            current_token = tokens[i].item()

            # Decode the current token to text and store it
            decoded_text = self.tokenizer.decode([current_token])
            decoded_texts.append(decoded_text)

            # Get crypto seed
            crypto_seed = self.rng.rand_index(
                self.rng.get_seed([prev_K_tokens], [index]), 0
            )
            # print(crypto_seed, index)

            # Reproduce the proposal step
            seed = int(crypto_seed[0].item() * 1145141919810)
            # print(prev_K_tokens, seed)
            proposal_token, proposal_probs = self.rng.proposal([prev_K_tokens], seed, temperature=self.proposal_temp)

            if (current_token, tuple(prev_values)) in seen:
                continue

            seen.add((current_token, tuple(prev_values)))

            # Record the probabilities and matching status
            match = current_token == proposal_token[0]
            cumul.append(1 if match else 0)
            cumul_probs.append(proposal_probs[0, proposal_token[0]].item())

        if not len(cumul):
            return [(False, 0.5, 0.5, 0, 0)]

        # Calculate the p-values
        results = []
        sum_count_i = 0
        sum_probs_i = 0

        for i in range(len(cumul)):
            sum_count_i += cumul[i]
            sum_probs_i += cumul_probs[i]

            if sum_probs_i == 0 or sum_count_i <= sum_probs_i:
                epsilon_i = torch.tensor(float('inf'))  # Avoid division by zero
                p_value = torch.tensor(1)  # If sum_probs_i is 0, set p_value to 1
            else:
                epsilon_i = torch.tensor(sum_count_i / sum_probs_i - 1)
                sum_probs_i_tensor = torch.tensor(sum_probs_i)
                p_value = torch.exp((epsilon_i - (1 + epsilon_i) * torch.log(1 + epsilon_i)) * sum_probs_i_tensor) # Following Bernoulli variable's concentration inequality

            p_values.append(p_value.item())
            watermarked = p_value < 0.05
            cnt = i + 1
            results.append((watermarked.item(), sum_count_i/cnt, p_value.item(), cnt, i))

        # Write the results to a TSV file
        # with open("results_debug/verification_results.tsv", "w") as tsv_file:
        #     tsv_file.write("Index\tText\tMatches\tProposal_Probability\tP_Value\n")
        #     for i in range(len(cumul)):
        #         match_status = "Yes" if cumul[i] == 1 else "No"
        #         # Write each line to the TSV file
        #         tsv_file.write(f"{i}\t{decoded_texts[i]}\t{match_status}\t{cumul_probs[i]:.6f}\t{p_values[i]:.6f}\n")

        # print(results, cumul_probs, cumul)

        return results



class BootstrapEmpiricalVerifier(EmpiricalVerifier):
    """
    A class for verifying the distribution shift of a watermark using empirical testing.

    Inherits from EmpiricalVerifier.

    Args:
        rng (RandomNumberGenerator): A random number generator object.
        pvalue (float): The p-value threshold for the statistical test.
        tokenizer (Tokenizer): A tokenizer object.
        method (str): The method used to generate the watermark.
        gamma_watermark (float): The gamma value for the watermark.
        gamma_edit_distance (float): The gamma value for the edit distance.

    Methods:
        score_matrix(tokens, random_values, index=0): Computes the score matrix for the given tokens and random values.
        random_score_matrix(tokens, random_shape, shared_randomness, index=0): Produces a random score matrix.
    """

    def __init__(
        self,
        rng,
        pvalue,
        tokenizer,
        method,
        gamma_watermark,
        gamma_edit_distance,
    ):
        super().__init__(
            rng, pvalue, tokenizer, method, gamma_edit_distance, False
        )
        self.gamma = gamma_watermark
        self.rand_size = 1

    def score_matrix(self, tokens, random_values, index=0, meta=None):
        _, L, _ = random_values.shape
        random_values = random_values[0, :, 0].reshape(1, L).cpu()

        tokens = tokens.squeeze().to(self.rng.device)
        if not tokens.nelement():
            return None

        greenlists = torch.stack(
            [
                self.rng.green_list(
                    random_values[:, i], self.gamma, True
                ).squeeze()
                for i in range(L)
            ]
        )
        greenlists = greenlists.repeat(1 + L // len(tokens), 1)[
            : len(tokens), :
        ].to(self.rng.device)
        rtn = 1 - (greenlists[:, tokens].float())
        return rtn.float()

    def random_score_matrix(
        self, tokens, random_shape, shared_randomness, index=0, meta=None
    ):
        """Produce a random score vector (faster to directly sample the random scores than to sample all random values)"""
        _, L, _ = random_shape
        val = (
            torch.cuda.FloatTensor(
                L, self.rng.vocab_size, device=self.rng.device
            )
            .uniform_(0, 1)[shared_randomness, :]
            .lt(self.gamma)
        )
        return 1 - (val[:, tokens.squeeze().to(self.rng.device)].float())

        # random_values = torch.rand((1,L), dtype=torch.float32).to(self.rng.device)
        # tokens = tokens.squeeze()
        # greenlists = [set(self.rng.green_list(random_values[:, i%L], self.gamma).squeeze().cpu().numpy()) for i in range(len(tokens))]
        # return torch.tensor([[0 if t.item() in g else 1 for t in tokens] for g in greenlists]).to(self.rng.device)

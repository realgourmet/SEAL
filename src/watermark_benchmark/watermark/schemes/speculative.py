import scipy
import torch

from watermark_benchmark.watermark.templates.generator import Watermark
from watermark_benchmark.watermark.templates.verifier import (
    EmpiricalVerifier,
    Verifier,
)


class SpeculativeGeneration(Watermark):
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

    def __init__(self, rng, verifier, tokenizer, temp, delta, gamma, K, proposal_temp, version='base', inputs='full', output_dir='', seed_factor=1145141919810):
        super().__init__(rng, verifier, tokenizer, temp)
        self.delta = delta
        self.gamma = gamma
        self.temp = temp
        self.K = K
        self.proposal_temp = proposal_temp
        self.version = version
        self.inputs = inputs
        self.output_dir = output_dir
        self.seed_factor = seed_factor

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

        if self.version == 'pro':
            return self.process_pro(logits, previous_tokens, ids)

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
        crypto_seeds = self.rng.rand_index(self.rng.get_seed(previous_K_tokens, ids), 0)

        # Sample a random hash function
        greenlist = self.rng.green_list(crypto_seeds, self.gamma)

        # Compute the probability of the original model
        p = torch.nn.functional.softmax(logits/self.temp, dim=-1).to(torch.float32)

        if self.version == 'lite':
            # Set proposal_probs to be a uniform distribution in lite version
            q = torch.full((N, self.rng.vocab_size), 1.0 / self.rng.vocab_size, device=p.device)
            proposal_set = greenlist

        elif self.version == 'base':
            # Get seed for the smaller model and proposal logits
            seed = int(crypto_seeds[0].item() * self.seed_factor) 
            proposal_token, q = self.rng.proposal(previous_K_tokens, seed, temperature=self.proposal_temp)
            q = q.to(p.device)

            # Initialize q_new with zeros
            q_new = torch.zeros_like(q)

            # Determine the proposal set
            proposal_set = []
            for i in range(N):
                # Identify the set that the proposal token is in
                if proposal_token[i] in greenlist[i]:
                    proposal_set.append(greenlist[i])  # Proposal set is the green list
                else:
                    # Proposal set is the complement of the green list
                    complement_indices = torch.tensor([j for j in range(q.size(1)) if j not in greenlist[i]], device=greenlist.device)
                    proposal_set.append(complement_indices)

        # Setup the acceptance probability set-wise
        q_new = torch.zeros_like(q).to(p.device)
        for i in range(N):
            proposal_set_indices = proposal_set[i]
            p_proposal_set = p[i, proposal_set_indices].sum()
            q_proposal_set = q[i, proposal_set_indices].sum()
            
            if p_proposal_set >= q_proposal_set:
                q_new[i, proposal_set_indices] = 1
            else:
                # Update the complement of the proposal set
                complement_indices = torch.tensor([j for j in range(q.size(1)) if j not in proposal_set_indices], device=p.device)
                q_new[i, complement_indices] = (1 - p_proposal_set / q_proposal_set.clamp(min=1e-12)).clamp(min=1e-12,max=1)
                q_new[i, proposal_set_indices] = p_proposal_set / q_proposal_set.clamp(min=1e-12)

        # Adjust q_new as p conditional on proposal_set and complement of proposal_set
        for i in range(N):
            # Normalize q_new within proposal set and complement
            proposal_indices = proposal_set[i]
            complement_indices = torch.tensor([j for j in range(q.size(1)) if j not in proposal_indices], device=p.device)

            q_new[i, proposal_indices] = (p[i, proposal_indices] / p[i, proposal_indices].sum().clamp(min=1e-12)) * q_new[i, proposal_indices]
            q_new[i, complement_indices] = (p[i, complement_indices] / p[i, complement_indices].sum().clamp(min=1e-12)) * q_new[i, complement_indices]

        # Convert q_new back to logits
        q_new_logits = torch.log(q_new + 1e-12) * self.temp

        # Add bias delta to the logits in the proposal set
        for i in range(N):
            # Ensure proposal_set[i] is a tensor
            proposal_indices = proposal_set[i] if isinstance(proposal_set[i], torch.Tensor) else torch.tensor(proposal_set[i], device=logits.device)
            q_new_logits[i, proposal_indices] += self.delta

        return q_new_logits

    # Generalized process function
    def process_pro(self, logits, previous_tokens, ids):
        """
        Applies the watermarking scheme to the input logits using multiple sets.

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
            return logits
        previous_K_tokens = [tokens[-self.K:] for tokens in previous_tokens]

        # Get crypto seed
        crypto_seeds = self.rng.rand_index(self.rng.get_seed(previous_K_tokens, ids), 0)
        
        # Adjust logits using self.temp (temperature)
        logits = logits / self.temp

        # Compute the probability of the original model
        p = torch.nn.functional.softmax(logits, dim=-1).to(torch.float32)

        # Get seed for the smaller model and proposal logits
        proposal_token, q = self.rng.proposal(previous_K_tokens, crypto_seeds, temperature=self.proposal_temp)
        q = q.to(p.device)

        # Partition the vocabulary into multiple sets
        partitions = self.rng.partition_vocab(crypto_seeds, self.gamma)

        # Initialize q_new and q_new_logits with zeros
        q_new = torch.zeros_like(q)
        q_new_logits = torch.zeros_like(q)

        # Determine the proposal set and compute probabilities
        proposal_set_indices = []
        for i in range(N):
            # Identify which partition contains the proposal token
            proposal_set = None
            proposal_set_index = -1
            for idx, partition in enumerate(partitions):
                if proposal_token[i] in partition[i]:
                    proposal_set = partition[i]
                    proposal_set_index = idx
                    proposal_set_indices.append(idx)
                    break

            # Calculate probabilities for each partition
            p_partition_probs = torch.tensor([p[i, partition[i]].sum().clamp(min=1e-12) for partition in partitions], device = p.device).squeeze()
            q_partition_probs = torch.tensor([q[i, partition[i]].sum().clamp(min=1e-12) for partition in partitions], device = p.device).squeeze()

            # Set acceptance probability
            if p_partition_probs[proposal_set_index] >= q_partition_probs[proposal_set_index]:
                # Accept the proposal set with probability 1
                q_new[i, proposal_set] = 1
            else:
                # Compute new probabilities for each partition
                for idx, partition in enumerate(partitions):
                    if idx == proposal_set_index:
                        # Set the probability for the proposal set
                        q_new[i, partition[i]] = p_partition_probs[idx] / q_partition_probs[idx]
                    else:
                        # Adjust other partitions' probabilities
                        q_new[i, partition[i]] = (1 - p_partition_probs[proposal_set_index] / q_partition_probs[proposal_set_index]) * \
                                                    (p_partition_probs[idx] - q_partition_probs[idx]).clamp(min=0) / \
                                                    (p_partition_probs - q_partition_probs).clamp(min=1e-12).sum()

            # Normalize q_new within each partition to ensure the sum of probabilities is 1
            for partition in partitions:
                partition_indices = partition[i]
                q_new[i, partition_indices] = (p[i, partition_indices] / p[i, partition_indices].sum().clamp(min=1e-12)) * q_new[i, partition_indices]

            # Add bias delta to the logits in the proposal set
            q_new_logits[i] = torch.log(q_new[i] + 1e-12) * self.temp
            q_new_logits[i, proposal_set] += self.delta

        return q_new_logits


class SpeculativeVerifier(Verifier):
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

    def __init__(self, rng, pvalue, tokenizer, gamma, K, proposal_temp, version='base', inputs='full', output_dir='', seed_factor=1145141919810):
        super().__init__(rng, pvalue, tokenizer)
        self.gamma = gamma
        self.K = K
        self.proposal_temp = proposal_temp
        self.version = version
        self.inputs = inputs
        self.output_dir = output_dir
        self.seed_factor = seed_factor

    def verify(self, tokens, index=0, exact=False, meta=None):
        if self.version == 'pro':
            return self.verify_pro(tokens, index, exact, meta)
        elif self.version == 'lite':
            i_start = 0
        elif self.version == 'base':
            i_start = self.K


        tokens = tokens.squeeze()
        cumul = []
        cumul_probs = []
        p_values = []
        seen = set()
        decoded_texts = []

        if not tokens.nelement() or not len(tokens.shape):
            return [(False, 0.5, 0.5, 0, 0)]

        for i in range(i_start, len(tokens)):
            prev_values = tokens[:i].tolist()

            current_token = tokens[i].item()

            # Decode the current token to text and store it
            decoded_text = '' # self.tokenizer.decode([current_token])
            decoded_texts.append(decoded_text)

            # Get crypto seed
            crypto_seeds = self.rng.rand_index(
                self.rng.get_seed([prev_values], [index]), 0
            )
            print(f"index: {index}")

            # Sample a random hash function for the green list
            greenlist = self.rng.green_list(crypto_seeds, self.gamma)

            if self.version == 'lite':
                # Set proposal_probs to be a uniform distribution in the lite version
                proposal_probs = torch.full((1, self.rng.vocab_size), 1.0 / self.rng.vocab_size)
                proposal_set = greenlist[0]
                
            elif self.version == 'base':
                if i < self.K:
                    continue
                else:
                    prev_K_tokens = tokens[i-self.K:i].tolist()

                # Reproduce the proposal step
                seed = int(crypto_seeds[0].item() * self.seed_factor)
                proposal_token, proposal_probs = self.rng.proposal([prev_K_tokens], seed, temperature=self.proposal_temp)

                if (current_token, tuple(prev_values)) in seen:
                    continue

                seen.add((current_token, tuple(prev_values)))
                
                # Determine the proposal set
                if proposal_token[0] in greenlist[0]:
                    proposal_set = greenlist[0]  # Proposal set is the green list
                else:
                    # Proposal set is the complement of the green list
                    proposal_set = torch.tensor([j for j in range(proposal_probs.size(1)) if j not in greenlist[0]], device=greenlist.device)

            # Calculate set-wise probabilities
            p_proposal_set = proposal_probs[0, proposal_set].sum().item()

            # Determine if the current token is in the proposal set
            match = current_token in set(proposal_set.cpu().numpy())
            cumul.append(1 if match else 0)
            cumul_probs.append(p_proposal_set)

        if not len(cumul):
            return [(False, 0.5, 0.5, 0, 0)]

        # Calculate the p-values
        results = []
        sum_count_i = 0
        sum_probs_i = 0

        # Compute exact probabilities recursively
        exact_probs = self.compute_exact_probs(cumul_probs)

        for i in range(len(cumul)):
            sum_count_i += cumul[i]
            sum_probs_i += cumul_probs[i]

            if sum_probs_i == 0 or sum_count_i <= sum_probs_i:
                epsilon_i = torch.tensor(float('inf'))  # Avoid division by zero
                p_value = torch.tensor(1)  # If sum_probs_i is 0, set p_value to 1
            else:
                epsilon_i = torch.tensor(sum_count_i / sum_probs_i - 1)

                if self.version == 'lite':
                    p_value = scipy.stats.binomtest(sum_count_i, i+1, self.gamma, alternative="greater").pvalue

                elif self.version == 'base':
                    # Calculate the exact p-value using binomial probabilities
                    p_value = self.compute_exact_p_value(exact_probs, sum_count_i, i + 1)

                    # # Following Bernoulli variable's concentration inequality, which is loose
                    # p_value = torch.exp((epsilon_i - (1 + epsilon_i) * torch.log(1 + epsilon_i)) * sum_probs_i_tensor)

            # Store statistics to results
            p_values.append(p_value.item())
            watermarked = p_value < 0.05
            cnt = i + 1
            results.append((watermarked.item(), sum_count_i/cnt, p_value.item(), cnt, i))

        if self.inputs == 'debug':
            # Write the results to a TSV file
            with open(self.output_dir+"/verification_results.tsv", "w") as tsv_file:
                tsv_file.write("Index\tText\tMatches\tProposal_Probability\tP_Value\n")
                for i in range(len(cumul)):
                    match_status = "Yes" if cumul[i] == 1 else "No"
                    # Write each line to the TSV file
                    tsv_file.write(f"{i}\t{decoded_texts[i]}\t{match_status}\t{cumul_probs[i]:.6f}\t{p_values[i]:.6f}\n")

        return results

    def verify_pro(self, tokens, index=0, exact=False, meta=None):
        tokens = tokens.squeeze()
        cumul = []
        cumul_probs = []
        p_values = []
        seen = set()
        decoded_texts = []

        if not tokens.nelement() or not len(tokens.shape):
            return [(False, 0.5, 0.5, 0, 0)]
            
        for i in range(self.K, len(tokens)):
            prev_values = tokens[:i].tolist()

            if i < self.K:
                continue
            else:
                prev_K_tokens = tokens[i - self.K:i].tolist()

            current_token = tokens[i].item()

            # Decode the current token to text and store it
            decoded_text = self.tokenizer.decode([current_token])
            decoded_texts.append(decoded_text)

            # Get crypto seed
            crypto_seeds = self.rng.rand_index(self.rng.get_seed([prev_K_tokens], [index]), 0)

            # Partition the vocabulary into multiple sets
            partitions = self.rng.partition_vocab(crypto_seeds, self.gamma)

            # Reproduce the proposal step
            proposal_token, proposal_probs = self.rng.proposal([prev_K_tokens], crypto_seeds, temperature=self.proposal_temp)

            if (current_token, tuple(prev_values)) in seen:
                continue

            seen.add((current_token, tuple(prev_values)))

            # Determine the current set from the partitions
            current_set = None
            for partition in partitions:
                if current_token in partition[0]:  # Find which partition contains the current token
                    current_set = partition[0]
                    break

            # Calculate set-wise probabilities
            p_current_set = proposal_probs[0, current_set].sum().item()

            # Determine if the proposal token is in the current set
            match = proposal_token[0] in set(current_set.cpu().numpy())
            cumul.append(1 if match else 0)
            cumul_probs.append(p_current_set)

        if not len(cumul):
            return [(False, 0.5, 0.5, 0, 0)]

        # Calculate the p-values
        results = []
        sum_count_i = 0
        sum_probs_i = 0

        # Compute exact probabilities recursively
        exact_probs = self.compute_exact_probs(cumul_probs)

        for i in range(len(cumul)):
            sum_count_i += cumul[i]
            sum_probs_i += cumul_probs[i]

            if sum_probs_i == 0 or sum_count_i <= sum_probs_i:
                p_value = 1.0  # If sum_probs_i is 0, set p_value to 1
            else:
                # Calculate the exact p-value
                p_value = self.compute_exact_p_value(exact_probs, sum_count_i, i + 1)

            # Store statistics to results
            p_values.append(p_value)
            watermarked = p_value < 0.05
            cnt = i + 1
            results.append((watermarked, sum_count_i / cnt, p_value, cnt, i))

        if self.inputs == 'debug':
            # Write the results to a TSV file
            with open(self.output_dir+"/verification_results.tsv", "w") as tsv_file:
                tsv_file.write("Index\tText\tMatches\tProposal_Probability\tP_Value\n")
                for i in range(len(cumul)):
                    match_status = "Yes" if cumul[i] == 1 else "No"
                    # Write each line to the TSV file
                    tsv_file.write(f"{i}\t{decoded_texts[i]}\t{match_status}\t{cumul_probs[i]:.6f}\t{p_values[i]:.6f}\n")

        return results


    def compute_exact_probs(self, cumul_probs):
        """
        Compute the exact probabilities that cumul[i] = j for all 0 <= j <= i <= n
        using the recursive formula.

        Args:
            cumul_probs (list or torch.Tensor): The probabilities of each Bernoulli trial.

        Returns:
            torch.Tensor: A tensor of exact probabilities with shape (n+1, n+1).
        """
        n = len(cumul_probs)
        # Initialize exact_prob tensor with zeros
        exact_prob = torch.zeros((n + 1, n + 1), dtype=torch.float32)

        # Base cases
        exact_prob[1, 0] = 1 - cumul_probs[0]
        exact_prob[1, 1] = cumul_probs[0]

        # Fill the table using the recursive formula
        for i in range(1, n):  # Start from 1 because the base case is already set for i=1
            for j in range(0, i + 2):  # Compute up to i+1 for the next row
                if j > 0:
                    exact_prob[i + 1, j] += cumul_probs[i] * exact_prob[i, j - 1]
                exact_prob[i + 1, j] += (1 - cumul_probs[i]) * exact_prob[i, j]

        return exact_prob

    def compute_exact_p_value(self, exact_probs, observed_successes, num_trials):
        """
        Compute the exact p-value based on the exact probabilities table.

        Args:
            exact_probs (torch.Tensor): The table of exact probabilities.
            observed_successes (int): The number of observed successes.
            num_trials (int): The number of Bernoulli trials.

        Returns:
            float: The exact p-value.
        """
        # Sum the probabilities for outcomes as extreme or more extreme than observed_successes
        p_value = exact_probs[num_trials, observed_successes:].sum().item()  # Sum over the tail
        return p_value



class SpeculativeEmpiricalVerifier(EmpiricalVerifier):
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

from abc import ABC, abstractmethod
import random
import torch
import hash_cpp
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Randomness(ABC):
    """
    Abstract base class for generating random numbers using a secret key and devices.

    Args:
        secret_key (Union[str, List[str]]): A secret key or list of secret keys used for generating random numbers.
        devices (Union[torch.device, List[torch.device]]): A device or list of devices on which to generate random numbers.
        vocab_size (int): The size of the vocabulary used for generating random numbers.

    Attributes:
        secret_key (Union[str, List[str]]): A secret key or list of secret keys used for generating random numbers.
        devices (List[torch.device]): A list of devices on which to generate random numbers.
        device (torch.device): The device on which to generate random numbers.
        vocab_size (int): The size of the vocabulary used for generating random numbers.
        state (torch.Tensor): The current state of the random number generator.
        generator (torch.Generator): The generator used for generating random numbers.
    """
    @abstractmethod
    def __init__(self, secret_key, devices, vocab_size):
        self.secret_key = secret_key

        if type(devices) == list:
            self.devices = devices
            self.device = devices[0]
        else:
            self.device = devices
            self.devices = [devices]

        self.vocab_size = vocab_size
        self.state = None
        self.reset()
        self.set_permutation()

        self.generator = torch.Generator("cpu")

    def reset(self):
        """
        Resets the state of the watermark generator to all zeros.
        """
        l = 1 if type(self.secret_key) != list else len(self.secret_key)
        self.state = torch.zeros((l,)).long()

    def normalize_previous_values(self, previous_values):
        """
        Normalize the previous values by padding them with 1s to make them of equal length, and converting them to a tensor.

        Args:
            previous_values (list or torch.Tensor): The previous values to normalize.

        Returns:
            torch.Tensor: The normalized previous values as a tensor.
        """
        if not isinstance(previous_values, torch.Tensor):
            max_len = max(len(p) for p in previous_values) 
            previous_values = [[1 for _ in range(max_len - len(p))] + p for p in previous_values]
            previous_values = torch.tensor(previous_values)

        if len(previous_values.shape) == 1:
            previous_values = previous_values.unsqueeze(0)

        return previous_values

    def get_seed(self, previous_values, ids):
        self.state[ids] += 1

    @abstractmethod
    def rand_index(self, seeds, index, device=None):
        pass

    @abstractmethod
    def rand_range(self, seeds, length, device=None):
        pass

    def get_secret(self, offset):
        return self.secret_key[offset] if type(self.secret_key) == list else self.secret_key

    def set_permutation(self):
        if self.vocab_size < 2:
            return

        if type(self.secret_key) == list:
            shuf = [list(range(self.vocab_size)) for _ in range(len(self.secret_key))]
            for idx, key in enumerate(self.secret_key):
                random.Random(key).shuffle(shuf[idx])
            permutation= torch.tensor(shuf)
        else:
            shuf = list(range(self.vocab_size))
            random.Random(self.secret_key).shuffle(shuf)
            permutation= torch.tensor(shuf).unsqueeze(0)

        inv_permutation = torch.zeros_like(permutation)
        indices = torch.arange(permutation.shape[0]).repeat(permutation.shape[1],1).t()
        indices = torch.cat((indices.unsqueeze(2), permutation.unsqueeze(2)), dim=2)
        inv_permutation[indices[:,:,0], indices[:,:,1]] = torch.arange(self.vocab_size).repeat(permutation.shape[0],1)
        
        self.permutation = {device: permutation.to(device) for device in self.devices}
        self.inv_permutation = {device: inv_permutation.to(device) for device in self.devices}

    def get_permutation(self, device, inv=False):
        if device not in self.permutation:
            if type(device) == torch.device and device.index in self.permutation:
                device = device.index
            else:
                print("Device not initialized for random number generator. The sampling procedure is occuring on device {}, while only {} are available. Copying over".format(device, self.devices))
                self.permutation[device] = self.permutation[self.device].to(device)
                self.inv_permutation[device] = self.inv_permutation[self.device].to(device)
        if inv:
            return self.inv_permutation[device]
        else:
            return self.permutation[device]


    def green_list(self, seeds, gamma,inv=False):
        gl_size = int(gamma*self.vocab_size)
        permutation = torch.cat( tuple(torch.randperm(self.vocab_size, generator=self.generator.manual_seed(int(h.item() * 2147483647))).unsqueeze(0) for h in seeds) )
        if not inv:
            return permutation[:, :gl_size]
        else:
            permutation = permutation.to(self.device)
            inv_permutation = torch.zeros_like(permutation)
            indices = torch.arange(permutation.shape[0], device=self.device).repeat(permutation.shape[1],1).t()
            indices = torch.cat((indices.unsqueeze(2), permutation.unsqueeze(2)), dim=2)
            inv_permutation[indices[:,:,0], indices[:,:,1]] = torch.arange(self.vocab_size,device=self.device).repeat(permutation.shape[0],1)
            return inv_permutation <= gl_size

    def partition_vocab(self, seeds, gamma):
        """
        Partition the vocabulary into multiple sets, each with a fraction gamma of the vocab size.

        Args:
            seeds (torch.Tensor): Seeds for generating random permutations.
            gamma (float): Fraction of the vocabulary size for each set.

        Returns:
            list of torch.Tensor: List of tensors, each representing a partition of the vocabulary.
        """
        partition_size = int(gamma * self.vocab_size)
        num_partitions = int(1 / gamma)
        
        # Ensure that gamma divides total vocab size
        assert partition_size * num_partitions == self.vocab_size, "Total size of partitions mismatch vocabulary size."

        # Create partitions
        partitions = []
        permutation = torch.cat(
            tuple(torch.randperm(self.vocab_size, generator=self.generator.manual_seed(int(h.item() * 2147483647))).unsqueeze(0) for h in seeds)
        )

        for i in range(num_partitions):
            start_index = i * partition_size
            end_index = start_index + partition_size
            partitions.append(permutation[:, start_index:end_index])

        return partitions



class EmbeddedRandomness(Randomness):
    """
    A class that represents embedded randomness.

    Args:
        Randomness (class): The base class for randomness.

    Attributes:
        hash_len (int): The length of the hash.
        min_hash (bool): A flag indicating whether to use the minimum hash.

    Methods:
        get_seed(previous_values, ids=None): Returns the seed for the given previous values and IDs.
        rand_range(seeds, length, device=None): Returns a random value for each index in a range for the given seeds range length.
        rand_index(seeds, index, device=None): Returns a random value for the given seeds and index.
    """
    def __init__(self, secret_key, device, vocab_size, hash_len, min_hash):
        super().__init__(secret_key, device, vocab_size)
        self.hash_len = hash_len
        self.min_hash = min_hash


    def get_seed(self, previous_values, ids=None):
        previous_values = self.normalize_previous_values(previous_values)
        N, _ = previous_values.shape
        if ids is None:
            ids = [0 for _ in range(N)]

        if not self.hash_len: # R1
            tmp = [[] for _ in range(previous_values.shape[0])]
        else:
            tmp = [[v.item() for v in prev[-self.hash_len:]] for prev in previous_values]
            tmp = [[-1 for _ in range(self.hash_len - len(value))] + value for value in tmp]

        if self.min_hash: # R2
            h = [str( round(min(hash_cpp.index_hash(["{}SEED{}".format(t, self.get_secret(ids[k]))], 0).cpu().item() for t in (tmp[k] if len(tmp[k]) else [0]) ), 8)) for k in range(N)]
        else:
            tmp = ["_".join(str(i) for i in t) if len(t) else "" for t in tmp]
            h = ["{}SEED{}".format(t, self.get_secret(ids[k])) for k,t in enumerate(tmp)]

        super().get_seed(previous_values, ids)
        return h


    def rand_range(self, seeds, length, device=None):
        if length == 0:
            length = self.vocab_size
        return hash_cpp.all_index_hash(seeds, torch.zeros((len(seeds), length), dtype=torch.float32).to(self.device if device is None else device))

    
    def rand_index(self, seeds, index, device=None):
        return hash_cpp.index_hash(seeds, index).to(self.device  if device is None else device)


class ExternalRandomness(Randomness):
    """
    A class representing an external source of randomness for generating watermarks.

    Args:
        secret_key (Union[str, List[str]]): The secret key used to initialize the random number generator(s).
        device (torch.device): The device to use for generating random numbers.
        vocab_size (int): The size of the vocabulary.
        key_len (int, optional): The length of the secret key. Defaults to 512.
        random_size (int, optional): The size of the random numbers to generate. Defaults to None.
    """
    def __init__(self, secret_key, device, vocab_size, key_len=512, random_size = None):
        self.key_len = key_len
        super().__init__(secret_key, device, vocab_size)

        self.rng = [random.Random(self.secret_key)] if type(self.secret_key) != list else [random.Random(key) for key in self.secret_key]

        if random_size is None:
            random_size = vocab_size

        self.random_size = random_size
        
        self.xi = torch.tensor([[r.random() for _ in range(self.key_len*self.random_size)] for r in self.rng], dtype=torch.float32).reshape(len(self.rng), self.key_len, self.random_size)


    def reset(self):
        super().reset()
        l = 1 if type(self.secret_key) != list else len(self.secret_key)
        self.shift = torch.randint(self.key_len, (l,))
        #self.shift = torch.zeros((l,)).long().to(self.device)


    def get_seed(self, previous_values, ids=None):
        previous_values = self.normalize_previous_values(previous_values)
        N, _ = previous_values.shape
        if ids is None:
            ids = torch.zeros((N,)).long()
        elif type(ids) != torch.Tensor:
            ids = torch.Tensor(ids).long()
        super().get_seed(previous_values, ids)

        rtn = torch.cat((ids.unsqueeze(0), ((self.shift[ids] + self.state[ids] - 1)%self.key_len).unsqueeze(0)), axis=0).t()
        return rtn
        

    def rand_range(self, index, length, device=None):
        if length:
            return self.xi[index[:,0], index[:,1], :length].to(self.device if device is None else device)
        else:
            return self.xi[index[:,0], index[:,1], :].to(self.device if device is None else device)

    
    def rand_index(self, index, token_index, device=None):
        return self.xi[index[:,0], index[:,1], token_index].to(self.device if device is None else device)


class BootstrapRandomness(Randomness):
    """
    A class representing an external source of randomness for generating watermarks.

    Args:
        secret_key (Union[str, List[str]]): The secret key used to initialize the random number generator(s).
        device (torch.device): The device to use for generating random numbers.
        vocab_size (int): The size of the vocabulary.
        key_len (int, optional): The length of the secret key. Defaults to 512.
        random_size (int, optional): The size of the random numbers to generate. Defaults to None.
    """
    def __init__(self, secret_key, device, vocab_size, key_len=512, random_size = None, model_device = 'cpu'):
        self.key_len = key_len
        super().__init__(secret_key, device, vocab_size)

        self.rng = [random.Random(self.secret_key)] if type(self.secret_key) != list else [random.Random(key) for key in self.secret_key]

        if random_size is None:
            random_size = vocab_size

        self.random_size = random_size
        
        self.xi = torch.tensor([[r.random() for _ in range(self.key_len*self.random_size)] for r in self.rng], dtype=torch.float32).reshape(len(self.rng), self.key_len, self.random_size)

        # Use Phi-3 3B as the smaller (proposal) model
        self.model_device = model_device
        # self.model_name='microsoft/Phi-3-mini-128k-instruct' # 'meta-llama/Llama-2-7b-chat-hf'

        self.model_name='openlm-research/open_llama_3b_v2' # 'meta-llama/Llama-2-7b-chat-hf'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True).to(self.model_device)
        self.model.eval()

        self.token_compatibility()

    def token_compatibility(self):
        """
        Check if the token IDs 0-31999 correspond to the same token in both the microsoft/Phi-3-mini-128k-instruct and meta-llama/Llama-2-7b-chat-hf models
        """
        
        # Load the tokenizers
        phi_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

        # Compare tokens by ID
        same_tokens = True
        for token_id in range(32000):
            phi_token = phi_tokenizer.convert_ids_to_tokens(token_id)
            llama_token = llama_tokenizer.convert_ids_to_tokens(token_id)
            
            if phi_token != llama_token:
                same_tokens = False
                print(f"Token ID {token_id} differs: Phi-3 -> {phi_token}, Llama -> {llama_token}")
                break

        if same_tokens:
            print("All tokens match for the first 32,000 token IDs.")
        else:
            print("Tokens do not match.")


    def reset(self):
        super().reset()
        l = 1 if type(self.secret_key) != list else len(self.secret_key)
        self.shift = torch.randint(self.key_len, (l,))
        #self.shift = torch.zeros((l,)).long().to(self.device)


    def get_seed(self, previous_values, ids=None):
        previous_values = self.normalize_previous_values(previous_values)
        N, _ = previous_values.shape
        if ids is None:
            ids = torch.zeros((N,)).long()
        elif type(ids) != torch.Tensor:
            ids = torch.Tensor(ids).long()
        super().get_seed(previous_values, ids)

        rtn = torch.cat((ids.unsqueeze(0), ((self.shift[ids] + self.state[ids] - 1)%self.key_len).unsqueeze(0)), axis=0).t()
        return rtn
        

    def rand_range(self, index, length, device=None):
        if length:
            return self.xi[index[:,0], index[:,1], :length].to(self.device if device is None else device)
        else:
            return self.xi[index[:,0], index[:,1], :].to(self.device if device is None else device)

    
    def rand_index(self, index, token_index, device=None):
        return self.xi[index[:,0], index[:,1], token_index].to(self.device if device is None else device)

    def proposal(self, previous_tokens, seed, temperature=2):
        # Set seeds manually
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        input_ids = torch.tensor(previous_tokens)
        input_ids = input_ids.to(self.model_device)
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :32000] # Phi-3 has 32064 tokens and Llama-2 has 32000 tokens
        
        # Apply temperature and find probability of the small model
        logits = logits / temperature
        logits = logits.clamp(min=-50, max=50)
        logits = logits - logits.max()  # Normalize logits to prevent overflow
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # Sample the next token from the probability distribution
        next_token = torch.multinomial(probabilities, num_samples=1).squeeze(1)
        next_token_prob = probabilities[torch.arange(logits.size(0)), next_token]
        # print(f"probability of the proposal token in smaller model: {next_token_prob}")

        return next_token.tolist(), probabilities


class BootstrapEmbeddedRandomness(EmbeddedRandomness):
    """
    A class inherited from EmbeddedRandomness.

    Args:
        Randomness (class): The base class for randomness.

    Attributes:
        hash_len (int): The length of the hash.
        min_hash (bool): A flag indicating whether to use the minimum hash.

    Methods:
        get_seed(previous_values, ids=None): Returns the seed for the given previous values and IDs.
        rand_range(seeds, length, device=None): Returns a random value for each index in a range for the given seeds range length.
        rand_index(seeds, index, device=None): Returns a random value for the given seeds and index.
    """
    def __init__(self, secret_key, device, vocab_size, hash_len, min_hash, model_device='cpu', proposal_model='microsoft/Phi-3-mini-128k-instruct', seed_factor=1145141919810):
        super().__init__(secret_key, device, vocab_size, hash_len, min_hash)

        # Use Phi-3 3B as the smaller (proposal) model
        self.model_device = model_device
        self.seed_factor = seed_factor
        
        self.model_name = proposal_model # 'lmsys/vicuna-7b-v1.5' # 'meta-llama/Llama-2-7b-chat-hf'

        print(f'Proposal model: {self.model_name}')

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, 
                        trust_remote_code=True,
                        # attn_implementation="flash_attention_2",
                        # torch_dtype=torch.bfloat16,
                    ).to(self.model_device)
        # self.model.generation_config.cache_implementation = "static"
        # self.model.forward = torch.compile(self.model.forward, mode="reduce-overhead", fullgraph=True)
        self.model.eval()

        self.token_compatibility()

    def token_compatibility(self):
        """
        Check if the token IDs 0-31999 correspond to the same token in both the microsoft/Phi-3-mini-128k-instruct and meta-llama/Llama-2-7b-chat-hf models
        """
        
        # Load the tokenizers
        phi_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

        # Compare tokens by ID
        same_tokens = True
        for token_id in range(32000):
            phi_token = phi_tokenizer.convert_ids_to_tokens(token_id)
            llama_token = llama_tokenizer.convert_ids_to_tokens(token_id)
            
            if phi_token != llama_token:
                same_tokens = False
                print(f"Token ID {token_id} differs: Phi-3 -> {phi_token}, Llama -> {llama_token}")
                break

        if same_tokens:
            print("All tokens match for the first 32,000 token IDs.")
        else:
            print("Tokens do not match.")

    def proposal(self, previous_tokens, seeds, temperature=2):
        """
        Sample the proposal tokens and return the proposal probabilities.
        """

        input_ids = torch.tensor(previous_tokens)
        input_ids = input_ids.to(self.model_device)
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :32000] # Phi-3 has 32064 tokens and Llama-2 has 32000 tokens
        
        # Apply temperature and find probability of the small model
        logits = logits / temperature
        logits = logits.clamp(min=-50, max=50)
        logits = logits - logits.max()  # Normalize logits to prevent overflow
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # Sample the next token using the manual seeds
        g = torch.Generator(device=self.model_device)
        next_tokens = []
        for idx, seed in enumerate(seeds):
            g.manual_seed(int(seed.item() * self.seed_factor))
            selected_token = torch.multinomial(
                    probabilities[idx], num_samples=1, generator=g
                )
            next_tokens.append(selected_token.item())

        # print(next_tokens, probabilities)

        return next_tokens, probabilities

import jax
from ngclearn.utils.model_utils import scanner
from ngcsimlib.compilers import compile_command, wrap_command
from ngcsimlib.context import Context
from ngclearn.utils import JaxProcess
from ngclearn.utils.io_utils import makedir
from jax import numpy as jnp, random, jit
from ngclearn.components import GaussianErrorCell as ErrorCell, RateCell, HebbianSynapse, StaticSynapse
import ngclearn.utils.weight_distribution as dist
from config import Config as config
from layers.embedding import EMBEDDING
from layers.attention import Attention
from utils.attention_utils import AttentionBlock
from utils.embed_utils import EmbeddingSynapse
from layers.mlp import MLP
from layers.output import Output

class NGCTransformer:
    """
    Predictive Coding Transformer following PCN architecture from:
    Whittington & Bogacz (2017) - "An approximation of the error backpropagation 
    algorithm in a predictive coding network with local hebbian synaptic plasticity"

    Architecture:
    z_embed -(W_embed)-> e_embed, z_qkv -(W_q,W_k,W_v - > W_attn_out)-> e_attn, z_mlp -(W_mlp1,W_mlp2)-> e_mlp, z_out -(W_out)-> e_out
    e_attn -(E_attn)-> z_qkv <- e_embed, e_mlp -(E_mlp)-> z_mlp <- e_attn, e_out -(E_out)-> z_out <- e_mlp

    Args:
        dkey: JAX seeding key
        vocab_size: vocabulary size
        seq_len: sequence length
        n_embed: embedding dimension
        n_heads: number of attention heads
        batch_size: batch size
        n_layers: number of transformer blocks
        dt: integration time constant
        tau_m: membrane time constant
        eta: learning rate for Hebbian synapses
        exp_dir: experimental directory
        model_name: unique model name
    """

    def __init__(self, dkey, target_ids, in_dim=1, out_dim=1, hid1_dim=128, hid2_dim=64, T=10,
                 dt=1., tau_m=10., act_fx="tanh", eta=0.001, exp_dir="exp",
                 model_name="pc_disc", loadDir=None, **kwargs):
        self.exp_dir = exp_dir
        self.model_name = model_name
        self.nodes = None
        makedir(exp_dir)
        makedir(exp_dir + "/filters")

        dkey, *subkeys = random.split(dkey, 30)

        self.T = T
        self.dt = dt
        optim_type = "adam"
        wlb = -0.3
        wub = 0.3
        gelu = jax.nn.gelu

        if loadDir is not None:
            self.load_from_disk(loadDir)
        else:
            with Context("Circuit") as self.circuit:
                
                self.embedding = EMBEDDING(dkey=subkeys[0])
                self.attention = Attention(dkey=subkeys[1])
                self.mlp = MLP(dkey=subkeys[2])
                self.output = Output(dkey=subkeys[3])
                self.z_target=RateCell("z_target", n_units=config.n_embed, tau_m=0., act_fx="identity", shape=(config.seq_len, config.n_embed, 1), batch_size=config.batch_size) 
                
                self.embedding.W_embed.inputs << self.embedding.z_embed.zF     
                self.embedding.e_embed.mu << self.embedding.W_embed.outputs   
                self.embedding.e_embed.target << self.attention.z_qkv.z
                

                 
                self.attention.W_q.inputs << self.attention.z_qkv.zF
                self.attention.W_k.inputs << self.attention.z_qkv.zF
                self.attention.W_v.inputs << self.attention.z_qkv.zF
                
                self.attention.attn_block.inputs_q << self.attention.W_q.outputs
                self.attention.attn_block.inputs_k << self.attention.W_k.outputs       
                self.attention.attn_block.inputs_v << self.attention.W_v.outputs
                
                self.attention.W_attn_out.inputs << self.attention.attn_block.outputs
                self.attention.e_attn.mu << self.attention.W_attn_out.outputs
                self.attention.e_attn.target << self.mlp.z_mlp.z

                self.mlp.W_mlp1.inputs << self.mlp.z_mlp.zF
                # self.mlp.e_mlp1.mu << self.mlp.W_mlp1.outputs
                # self.mlp.e_mlp.target << self.mlp.z_mlp2.z
                

                self.mlp.W_mlp2.inputs << self.mlp.W_mlp1.outputs
                self.mlp.e_mlp.mu << self.mlp.W_mlp2.outputs
                self.mlp.e_mlp.target << self.output.z_out.z
               
                self.output.W_out.inputs << self.output.z_out.zF
                self.output.e_out.mu << self.output.W_out.outputs
                self.output.e_out.target << self.z_target.z
                    

                                
                # self.embedding.E_embed.inputs << self.e_embed.dmu
                
                self.attention.E_attn.inputs << self.attention.e_attn.dmu
                
                self.mlp.E_mlp.inputs << self.mlp.e_mlp.dmu
                # self.E_mlp2.inputs << self.e_mlp2.dmu
                
                self.output.E_out.inputs << self.output.e_out.dmu
                
                
                # self.embedding.z_embed.j << self.embedding.E_embed.outputs
                # self.embedding.z_embed.j_td << td_error.e_attn.dtarget
                
                self.attention.z_qkv.j << self.attention.E_attn.outputs
                self.attention.z_qkv.j_td << self.embedding.e_embed.dtarget
                
                self.mlp.z_mlp.j << self.mlp.E_mlp.outputs
                self.mlp.z_mlp.j_td << self.attention.e_attn.dtarget
                # self.z_mlp2.j << self.E_mlp2.outputs
                # self.z_mlp2.j_td << self.e_mlp1.dtarget
                
                self.output.z_out.j << self.output.E_out.outputs
                self.output.z_out.j_td << self.mlp.e_mlp.dtarget
                
                
                # self.W_embed.pre << self.z_embed.zF
                self.embedding.W_embed.post << self.embedding.e_embed.dmu  
                self.attention.W_q.pre << self.attention.z_qkv.z
                self.attention.W_q.post << self.attention.e_attn.dmu
                
                self.attention.W_k.pre << self.attention.z_qkv.z
                self.attention.W_k.post << self.attention.e_attn.dmu
                
                self.attention.W_v.pre << self.attention.z_qkv.z
                self.attention.W_v.post << self.attention.e_attn.dmu
                self.attention.W_attn_out.pre << self.attention.attn_block.outputs
                self.attention.W_attn_out.post << self.attention.e_attn.dmu
                
                self.mlp.W_mlp1.pre << self.mlp.z_mlp.zF
                self.mlp.W_mlp1.post << self.mlp.e_mlp.dmu

                self.mlp.W_mlp2.pre << self.mlp.W_mlp1.outputs
                self.mlp.W_mlp2.post << self.mlp.e_mlp.dmu
               
                self.output.W_out.pre << self.output.z_out.zF
                self.output.W_out.post << self.output.e_out.dmu      
                
               
                self.q_embed = RateCell("q_embed", n_units=config.n_embed, tau_m=0., act_fx="identity",
                            shape=(config.seq_len, config.n_embed, 1), batch_size=config.batch_size)
                self.q_qkv = RateCell("q_qkv", n_units=config.n_embed, tau_m=1., act_fx="identity",
                          shape=(config.seq_len, config.n_embed, 1), batch_size=config.batch_size)
                self.q_mlp = RateCell("q_mlp", n_units=config.n_embed, tau_m=1., act_fx="identity",
                          shape=(config.seq_len, config.n_embed, 1), batch_size=config.batch_size)
                self.q_out = RateCell("q_out", n_units=config.n_embed, tau_m=1., act_fx="identity",
                          shape=(config.seq_len, config.n_embed, 1), batch_size=config.batch_size)
                self.q_target = RateCell("q_target", n_units=config.n_embed, tau_m=0., act_fx="identity",
                             shape=(config.seq_len, config.n_embed, 1), batch_size=config.batch_size)
                
                
                self.Q_embed = EmbeddingSynapse("Q_embed", vocab_size=config.vocab_size, seq_len=config.seq_len,
                                embed_dim=config.n_embed, batch_size=config.batch_size,
                                pos_learnable=config.pos_learnable, eta=config.eta,
                                optim_type=config.optim, key=subkeys[5])
                
                self.Q_q = StaticSynapse("Q_q", shape=(config.n_embed, config.n_embed), eta=config.eta,
                          weight_init=dist.uniform(amin=-0.3, amax=0.3), bias_init=dist.constant(value=0.),
                          w_bound=0., optim_type=config.optim, sign_value=-1., key=subkeys[6])
                
                self.Q_k = StaticSynapse("Q_k", shape=(config.n_embed, config.n_embed), eta=config.eta,
                          weight_init=dist.uniform(amin=-0.3, amax=0.3), bias_init=dist.constant(value=0.),
                          w_bound=0., optim_type=config.optim, sign_value=-1., key=subkeys[7])
                
                self.Q_v = StaticSynapse("Q_v", shape=(config.n_embed, config.n_embed), eta=config.eta,
                          weight_init=dist.uniform(amin=-0.3, amax=0.3), bias_init=dist.constant(value=0.),
                          w_bound=0., optim_type=config.optim, sign_value=-1., key=subkeys[8])
                
                self.Q_attn_out = StaticSynapse("Q_attn_out", shape=(config.n_embed, config.n_embed), eta=config.eta,
                                 weight_init=dist.uniform(amin=-0.3, amax=0.3), bias_init=dist.constant(value=0.),
                                 w_bound=0., optim_type=config.optim, sign_value=-1., key=subkeys[9])
                
                self.Q_mlp1 = StaticSynapse("Q_mlp1", shape=(4*config.n_embed, config.n_embed), eta=config.eta,
                              weight_init=dist.uniform(amin=-0.3, amax=0.3), bias_init=dist.constant(value=0.),
                              w_bound=0., optim_type=config.optim, sign_value=-1., key=subkeys[10])
                
                self.Q_mlp2 = StaticSynapse("Q_mlp2", shape=(config.n_embed, 4*config.n_embed), eta=config.eta,
                              weight_init=dist.uniform(amin=-0.3, amax=0.3), bias_init=dist.constant(value=0.),
                              w_bound=0., optim_type=config.optim, sign_value=-1., key=subkeys[11])
                
                self.Q_out = StaticSynapse("Q_out", shape=(config.vocab_size, config.n_embed), eta=config.eta,
                             weight_init=dist.uniform(amin=-0.3, amax=0.3), bias_init=dist.constant(value=0.),
                             w_bound=0., optim_type=config.optim, sign_value=-1., key=subkeys[12])
                
                self.eq_target = ErrorCell("eq_target", n_units=config.n_embed)
                
                
                self.Q_embed.inputs << self.q_embed.zF
                self.q_qkv.j << self.Q_embed.outputs
                self.Q_q.inputs << self.q_qkv.zF
                self.Q_k.inputs << self.q_qkv.zF
                self.Q_v.inputs << self.q_qkv.zF
                
                self.q_attn_block = AttentionBlock("q_attn_block", z_qkv=self.q_qkv, W_q=self.Q_q, W_k=self.Q_k, W_v=self.Q_v,
                                   n_heads=config.n_heads, n_embed=config.n_embed, seq_len=config.seq_len,
                                   dropout_rate=config.dropout_rate, batch_size=config.batch_size)
                
                self.Q_attn_out.inputs << self.q_attn_block.outputs
                self.q_mlp.j << self.Q_attn_out.outputs
                self.Q_mlp1.inputs << self.q_mlp.zF
                self.Q_mlp2.inputs << self.Q_mlp1.outputs
                self.q_out.j << self.Q_mlp2.outputs
                self.Q_out.inputs << self.q_out.zF
                self.q_target.j << self.Q_out.outputs
                self.eq_target.target << self.q_target.z 

                            
                
                advance_process = (JaxProcess(name="advance_process")
                                   >> self.attention.E_attn.advance_state
                                   >> self.mlp.E_mlp.advance_state
                                   >> self.output.E_out.advance_state
                                   >> self.embedding.z_embed.advance_state
                                   >> self.attention.z_qkv.advance_state
                                   >> self.mlp.z_mlp.advance_state
                                   >> self.output.z_out.advance_state
                                   >> self.embedding.W_embed.advance_state
                                   >> self.attention.W_q.advance_state
                                   >> self.attention.W_k.advance_state
                                   >> self.attention.W_v.advance_state
                                   >> self.attention.attn_block.advance_state
                                   >> self.attention.W_attn_out.advance_state
                                   >> self.mlp.W_mlp1.advance_state
                                   >> self.mlp.W_mlp2.advance_state
                                   >> self.output.W_out.advance_state
                                   >> self.embedding.e_embed.advance_state
                                   >> self.attention.e_attn.advance_state
                                   >> self.mlp.e_mlp.advance_state
                                   >> self.output.e_out.advance_state)

                reset_process = (JaxProcess(name="reset_process")
                                 >> self.q_embed.reset
                                 >> self.q_qkv.reset
                                 >> self.q_attn_block.reset
                                 >> self.q_mlp.reset
                                 >> self.q_out.reset
                                 >> self.q_target.reset
                                 >> self.eq_target.reset
                                 >> self.embedding.z_embed.reset
                                 >> self.attention.z_qkv.reset
                                 >> self.mlp.z_mlp.reset
                                 >> self.output.z_out.reset
                                 >> self.embedding.e_embed.reset
                                 >> self.attention.e_attn.reset
                                 >> self.mlp.e_mlp.reset
                                 >> self.output.e_out.reset)

                evolve_process = (JaxProcess(name="evolve_process")
                                  >> self.embedding.W_embed.evolve
                                  >> self.attention.W_q.evolve
                                  >> self.attention.W_k.evolve
                                  >> self.attention.W_v.evolve
                                  >> self.attention.W_attn_out.evolve
                                  >> self.mlp.W_mlp1.evolve
                                  >> self.mlp.W_mlp2.evolve
                                  >> self.output.W_out.evolve)

                project_process = (JaxProcess(name="project_process")
                                   >> self.q_embed.advance_state
                                   >> self.Q_embed.advance_state
                                   >> self.q_qkv.advance_state
                                   >> self.Q_q.advance_state
                                   >> self.Q_k.advance_state
                                   >> self.Q_v.advance_state
                                   >> self.q_attn_block.advance_state
                                   >> self.Q_attn_out.advance_state
                                   >> self.q_mlp.advance_state
                                   >> self.Q_mlp1.advance_state
                                   >> self.Q_mlp2.advance_state
                                   >> self.q_out.advance_state
                                   >> self.Q_out.advance_state
                                   >> self.q_target.advance_state
                                   >> self.eq_target.advance_state)

                processes = (reset_process, advance_process, evolve_process, project_process)        

                self._dynamic(processes)
    
    def _dynamic(self, processes):
        vars = self.circuit.get_components("q_embed", "q_qkv", "q_mlp", "q_out", 
                                           "q_target", "eq_target","Q_embed", "Q_q", "Q_k", "Q_v", "Q_attn_out",
                                           "Q_mlp1", "Q_mlp2", "Q_out",
                                           "z_embed", "z_qkv", "z_mlp", "z_out",
                                           "e_embed", "e_attn", "e_mlp", "e_out",
                                           "W_embed", "W_q", "W_k","W_v", "W_attn_out", 
                                           "W_mlp1", "W_mlp2", "W_out", "E_attn", "E_mlp", "E_out")
        (self.q_embed, self.q_qkv, self.q_mlp, self.q_out, 
        self.q_target, self.eq_target, self.Q_embed, self.Q_q, self.Q_k, self.Q_v, self.Q_attn_out,
        self.Q_mlp1, self.Q_mlp2, self.Q_out,
        self.embedding.z_embed, self.attention.z_qkv, self.mlp.z_mlp, self.output.z_out, self.embedding.e_embed, self.attention.e_attn, self.mlp.e_mlp, self.output.e_out, self.embedding.W_embed,
        self.attention.W_q, self.attention.W_k, self.attention.W_v, self.attention.W_attn_out,self.mlp.W_mlp1,self.mlp.W_mlp2, self.output.W_out, self.attention.E_attn, self.mlp.E_mlp, self.output.E_out) = vars
        self.nodes = vars

        reset_proc, advance_proc, evolve_proc, project_proc = processes

        self.circuit.wrap_and_add_command(jit(reset_proc.pure), name="reset")
        self.circuit.wrap_and_add_command(jit(advance_proc.pure), name="advance")
        self.circuit.wrap_and_add_command(jit(project_proc.pure), name="project")
        self.circuit.wrap_and_add_command(jit(evolve_proc.pure), name="evolve")

        @Context.dynamicCommand
        def clamp_input(x):
            self.embedding.z_embed.j.set(x)
            self.q_embed.j.set(x) 
        
        @Context.dynamicCommand
        def clamp_target(y):
            self.z_target.j.set(y)

        @Context.dynamicCommand
        def clamp_infer_target(y):
            self.eq_target.target.set(y)
        
    def save_to_disk(self, params_only=False):
        """
        Saves current model parameter values to disk

        Args:
            params_only: if True, save only param arrays to disk (and not JSON sim/model structure)
        """
        if params_only:
            model_dir = "{}/{}/custom".format(self.exp_dir, self.model_name)
            self.embedding.W_embed.save(model_dir)
            self.attention.W_q.save(model_dir)
            self.attention.W_k.save(model_dir)
            self.attention.W_v.save(model_dir)
            self.attention.W_attn_out.save(model_dir)
            self.mlp.W_mlp1.save(model_dir)
            self.mlp.W_mlp2.save(model_dir)     
            self.output.W_out.save(model_dir)
        else:
            self.circuit.save_to_json(self.exp_dir, model_name=self.model_name, overwrite=True)

    def load_from_disk(self, model_directory):
        """
        Loads parameter/config values from disk to this model

        Args:
            model_directory: directory/path to saved model parameter/config values
        """
        print(" > Loading model from ",model_directory)
        with Context("Circuit") as self.circuit:
            self.circuit.load_from_dir(model_directory)
            processes = (
                self.circuit.reset_process, self.circuit.advance_process,
                self.circuit.evolve_process, self.circuit.project_process
            )
            self._dynamic(processes)

    def process(self, obs, lab, adapt_synapses=True):
        eps = 0.001
        _lab = jnp.clip(lab, eps, 1. - eps)
        self.circuit.reset()

        ## pin/tie inference synapses to be exactly equal to the forward ones
        self.Q_embed.word_weights.set(self.embedding.W_embed.word_weights.value)
        if self.embedding.W_embed.pos_learnable:
           self.Q_embed.pos_weights.set(self.embedding.W_embed.pos_weights.value)
        self.Q_q.weights.set(self.attention.W_q.weights.value)
        self.Q_q.biases.set(self.attention.W_q.biases.value)
        self.Q_k.weights.set(self.attention.W_k.weights.value)
        self.Q_k.biases.set(self.attention.W_k.biases.value)
        self.Q_v.weights.set(self.attention.W_v.weights.value)
        self.Q_v.biases.set(self.attention.W_v.biases.value)
        self.Q_attn_out.weights.set(self.attention.W_attn_out.weights.value)
        self.Q_attn_out.biases.set(self.attention.W_attn_out.biases.value)
        self.Q_mlp1.weights.set(self.mlp.W_mlp1.weights.value)
        self.Q_mlp1.biases.set(self.mlp.W_mlp1.biases.value)
        self.Q_mlp2.weights.set(self.mlp.W_mlp2.weights.value)
        self.Q_mlp2.biases.set(self.mlp.W_mlp2.biases.value)
        self.Q_out.weights.set(self.output.W_out.weights.value)
        self.Q_out.biases.set(self.output.W_out.biases.value)
        
        ## pin/tie feedback synapses to transpose of forward ones
        self.attention.E_attn.weights.set(jnp.transpose(self.attention.W_attn_out.weights.value))
        self.mlp.E_mlp.weights.set(jnp.transpose(self.mlp.W_mlp2.weights.value))    
        self.output.E_out.weights.set(jnp.transpose(self.output.W_out.weights.value))
        # self.E_mlp2.weights.set(jnp.transpose(self.W2.weights.value))

        ## Perform P-step (projection step)
        self.circuit.clamp_input(obs)
        self.circuit.clamp_infer_target(_lab)
        self.circuit.project(t=0., dt=1.) 

        ## initialize dynamics of generative model latents to projected states for the errors it's 0
        self.attention.z_qkv.z.set(self.q_qkv.z.value)
        self.mlp.z_mlp.z.set(self.q_mlp.z.value)
        self.output.z_out.z.set(self.q_out.z.value)
       
        
        ## get projected prediction (from the P-step)
        y_mu_inf = self.q_target.z.value

        EFE = 0. ## expected free energy
        y_mu = 0.
        if adapt_synapses:
            for ts in range(0, self.T):
                self.circuit.clamp_input(obs) ## clamp input data to z_embed & q_embed input compartments
                self.circuit.clamp_target(_lab) ## clamp target data to z_target
                self.circuit.advance(t=ts, dt=1.)

            y_mu = self.output.e_out.mu.value ## get settled prediction
            ## calculate approximate EFE
            L1 = self.embedding.e_embed.L.value
            L2 = self.attention.e_attn.L.value
            L3 = self.mlp.e_mlp.L.value
            L4 = self.output.e_out.L.value
            EFE = L4 + L3 + L2 + L1

            if adapt_synapses == True:
                self.circuit.evolve(t=self.T, dt=1.)
                
        ## skip E/M steps if just doing test-time inference
        return y_mu_inf, y_mu, EFE

    def get_latents(self):
        return self.q_out.z.value

    def _get_norm_string(self): ## for debugging 
        _W_embed = self.embedding.W_embed.weights.value
        _W_q = self.attention.W_q.weights.value
        _W_k = self.attention.W_k.weights.value
        _W_v = self.attention.W_v.weights.value
        _W_attn_out = self.attention.W_attn_out.weights.value
        _W_mlp1 = self.mlp.W_mlp1.weights.value
        _W_mlp2 = self.mlp.W_mlp2.weights.value
        _W_out = self.output.W_out.weights.value
        _b_q = self.attention.W_q.biases.value
        _b_k = self.attention.W_k.biases.value
        _b_v = self.attention.W_v.biases.value
        _b_attn_out = self.attention.W_attn_out.biases.value
        _b_mlp1 = self.mlp.W_mlp1.biases.value
        _b_mlp2 = self.mlp.W_mlp2.biases.value
        _b_out = self.output.W_out.biases.value
        _norms = "_W_embed: {} _W_q: {} _W_k: {} _W_v: {} _W_attn_out: {} _W_mlp1: {} _W_mlp2: {} _W_out: {}\n _b_q: {} _b_k: {} _b_v: {} _b_attn_out:{} _b_mlp1: {} _b_mlp2 _b_out".format(jnp.linalg.norm(_W_embed),
                                                                      jnp.linalg.norm(_W_q),                                                          jnp.linalg.norm(_b_mlp2),
                                                                      jnp.linalg.norm(_b_out))
        return _norms
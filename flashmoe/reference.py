from .jit import ContextHandle, InitArgs, _get_compiled, ForwardArgs

def initialize(arg: InitArgs,) -> ContextHandle:
    from .bindings import reference_bindings
    mod_prefix = "flashmoe_reference"
    mod_name = (f"{mod_prefix}_s{arg.tokens_per_rank}_h{arg.token_dim}_i{arg.ffn_size}"
            f"_e{arg.num_experts}_ec{arg.expert_peer_capacity}_k{arg.top_k}"
            f"_topo{arg.topo}_mt{arg.mlp_type}_dt{arg.data_type}_act{arg.act_type}_arch{arg.gpu_arch}")
    src = reference_bindings.substitute(
        arch=arg.gpu_arch,
        s=arg.tokens_per_rank,
        h=arg.token_dim,
        i=arg.ffn_size,
        e=arg.num_experts,
        ec=arg.expert_peer_capacity,
        tk=arg.top_k,
        mod_name=mod_name,
        mt=arg.mlp_type,
        act=arg.act_type,
        dt=arg.data_type
    )
    mod = _get_compiled(arg, src, mod_prefix, mod_name)
    mod.initialize(device_id=arg.device_id)
    return ContextHandle(mod, None)

class RefForwardArgs:
    expert_up: int
    expert_down: int
    bias_up: int
    bias_down: int
    ref_input: int
    ref_interim0: int
    ref_interim1: int
    ref_out: int
    expert_up_v: int = 0
    bias_up_v: int = 0

    def __init__(self,
                 expert_up: int,
                 expert_down: int,
                 bias_up: int,
                 bias_down: int,
                 ref_input: int,
                 ref_interim0: int,
                 ref_interim1: int,
                 ref_out: int,
                 *,
                 expert_up_v: int,
                 bias_up_v: int):
        self.ref_input = ref_input
        self.ref_interim0 = ref_interim0
        self.ref_interim1 = ref_interim1
        self.ref_out = ref_out
        self.expert_up_v = expert_up_v
        self.bias_up = bias_up
        self.bias_down = bias_down
        self.bias_up_v = bias_up_v
        self.expert_up = expert_up
        self.expert_down = expert_down

def forward(handle: ContextHandle, token_ids: int, f_args: ForwardArgs, r_args: RefForwardArgs) -> None:
    handle.mod.forward(token_ids=token_ids,
                      tokens=f_args.tokens,
                      ref_input=r_args.ref_input,
                      expert_up=r_args.expert_up,
                      expert_up_v=r_args.expert_up_v,
                      expert_down=r_args.expert_down,
                      bias_up=r_args.bias_up,
                      bias_up_v=r_args.bias_up_v,
                      bias_down=r_args.bias_down,
                      ref_interim0=r_args.ref_interim0,
                      ref_interim1=r_args.ref_interim1,
                      expert_counts=f_args.expert_counts,
                      ref_out=r_args.ref_out,
                      swish_alpha=f_args.swish_alpha,
                      swish_beta=f_args.swish_beta,
                      stream_ptr=f_args.stream_ptr)


def parser_add_main_args(parser):
    # setup and protocol
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--ood_type', type=str, default='structure', choices=['structure', 'label', 'feature'],
                        help='only for cora/amazon/arxiv datasets')
    parser.add_argument('--data_dir', type=str, default='../../data/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--train_prop', type=float, default=.1,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.1,
                        help='validation label proportion')
    parser.add_argument('--runs', type=int, default=1, help='number of distinct runs')
    parser.add_argument('--epochs', type=int, default=200)

    # model network
    parser.add_argument('--method', type=str, default='msp', choices=['msp', 'gnnsafe'])
    parser.add_argument('--backbone', type=str, default='gcn')
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for GNN classifiers')
    parser.add_argument('--gat_heads', type=int, default=8,
                        help='attention heads for gat')
    parser.add_argument('--out_heads', type=int, default=1,
                        help='out heads for gat')
    parser.add_argument('--hops', type=int, default=2,
                        help='power of adjacency matrix for sgc')

    # gnnsafe hyper
    parser.add_argument('--T', type=float, default=1.0, help='temperature for Softmax')
    parser.add_argument('--use_reg', action='store_true', help='whether to use energy regularization loss')
    parser.add_argument('--lamda', type=float, default=1.0, help='weight for regularization')
    parser.add_argument('--m_in', type=float, default=-5, help='upper bound for in-distribution energy')
    parser.add_argument('--m_out', type=float, default=-1, help='lower bound for in-distribution energy')
    parser.add_argument('--use_prop', action='store_true', help='whether to use energy belief propagation')
    parser.add_argument('--K', type=int, default=2, help='number of layers for energy belief propagation')
    parser.add_argument('--alpha', type=float, default=0.5, help='weight for residual connection in propagation')

    # baseline hyper
    parser.add_argument('--noise', type=float, default=0., help='param for baseline ODIN and Mahalanobis')

    # training
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--use_bn', action='store_true', help='use batch norm')
    parser.add_argument('--sgcn_input_drop', type=float, default=0.0, help='input dropout for SGCN-style PyG backbones')
    parser.add_argument('--sgcn_edge_drop', type=float, default=0.0, help='edge dropout for SGCN-style PyG backbones')
    parser.add_argument('--sgcn_jk', action='store_true', help='enable simple jump-knowledge aggregation for SGCN-style PyG backbones')
    parser.add_argument('--sgcn_min_subgraph_nodes', type=int, default=0, help='minimum nodes to keep in each sampled SGCN subgraph')
    parser.add_argument('--sgcn_min_train_nodes', type=int, default=32, help='minimum train nodes to inject into each sampled SGCN subgraph')
    parser.add_argument(
        '--sgcn_energy_weighting',
        type=str,
        default='none',
        choices=['none', 'rank', 'sigmoid', 'hard'],
        help='subgraph-level SGCN reweighting driven by GNNSafe node energy',
    )
    parser.add_argument(
        '--sgcn_energy_aggregation',
        type=str,
        default='median',
        choices=['mean', 'median', 'trimmed_mean', 'bottomk_mean'],
        help='how to aggregate node energies into one subgraph score',
    )
    parser.add_argument('--sgcn_energy_trim_ratio', type=float, default=0.1, help='trim ratio for trimmed-mean subgraph energy aggregation')
    parser.add_argument('--sgcn_energy_bottomk_ratio', type=float, default=0.2, help='bottom-k ratio for bottom-k subgraph energy aggregation')
    parser.add_argument('--sgcn_energy_weight_min', type=float, default=0.5, help='minimum soft weight for subgraph energy reweighting')
    parser.add_argument('--sgcn_energy_weight_max', type=float, default=1.5, help='maximum soft weight for subgraph energy reweighting')
    parser.add_argument('--sgcn_energy_sigmoid_tau', type=float, default=1.0, help='temperature for sigmoid-based subgraph energy reweighting')
    parser.add_argument('--sgcn_energy_hard_keep_ratio', type=float, default=0.5, help='fraction of low-energy subgraphs kept by hard filtering')
    parser.add_argument('--sgcn_energy_scorer_epochs', type=int, default=100, help='training epochs for the frozen GNNSafe energy scorer')

    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=10, help='how often to print')
    parser.add_argument('--eval_step', type=int,
                        default=10, help='how often to run full evaluation')
    parser.add_argument('--cached', action='store_true',
                        help='set to use faster sgc')
    parser.add_argument('--print_prop', action='store_true',
                        help='print proportions of predicted class')
    parser.add_argument('--print_args', action='store_true',
                        help='print args for hyper-parameter searching')
    parser.add_argument('--mode', type=str, default='detect', choices=['classify', 'detect'])

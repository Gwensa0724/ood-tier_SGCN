#!/usr/bin/env bash
set -e

dev=${1:-0}
data_dir=${2:-../../data/}

echo "device: ${dev}"
echo "data_dir: ${data_dir}"

# PubMed OOD detection runs
python main.py --method msp --backbone gcn --dataset pubmed --ood_type structure --mode detect --use_bn --device ${dev} --data_dir ${data_dir}
python main.py --method gnnsafe --backbone gcn --dataset pubmed --ood_type structure --mode detect --use_bn --device ${dev} --data_dir ${data_dir}
python main.py --method gnnsafe --backbone gcn --dataset pubmed --ood_type structure --mode detect --use_bn --use_reg --m_in -5 --m_out -1 --lamda 0.01 --device ${dev} --data_dir ${data_dir}
python main.py --method gnnsafe --backbone gcn --dataset pubmed --ood_type structure --mode detect --use_bn --use_prop --device ${dev} --data_dir ${data_dir}
python main.py --method gnnsafe --backbone gcn --dataset pubmed --ood_type structure --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda 0.01 --device ${dev} --data_dir ${data_dir}

# Export energy scores for visualization
python discuss.py --dis_type vis_energy --method gnnsafe --backbone gcn --dataset pubmed --ood_type structure --mode detect --use_bn --device ${dev} --data_dir ${data_dir}
python discuss.py --dis_type vis_energy --method gnnsafe --backbone gcn --dataset pubmed --ood_type structure --mode detect --use_bn --use_reg --m_in -5 --m_out -1 --lamda 0.01 --device ${dev} --data_dir ${data_dir}
python discuss.py --dis_type vis_energy --method gnnsafe --backbone gcn --dataset pubmed --ood_type structure --mode detect --use_bn --use_prop --device ${dev} --data_dir ${data_dir}
python discuss.py --dis_type vis_energy --method gnnsafe --backbone gcn --dataset pubmed --ood_type structure --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda 0.01 --device ${dev} --data_dir ${data_dir}

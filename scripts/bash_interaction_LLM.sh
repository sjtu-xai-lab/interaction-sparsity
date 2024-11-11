python demo/sparsify_opt_train.py --save_dir_name="interaction_opt_reproduce" --gpu_id=1 --lr=1e-5 --epochs=20000 --mode="q"
python demo/sparsify_aquila_train.py --save_dir_name="interaction_aquila_reproduce" --gpu_id=4 --lr=1e-5 --epochs=20000 --mode="q"
python demo/sparsify_llama_train.py --save_dir_name="interaction_llama_reproduce" --gpu_id=1 --lr=1e-5 --epochs=20000 --mode="q"

python demo/sparsify_opt_train.py --save_dir_name="interaction_opt_reproduce" --gpu_id=1 --lr=1e-6 --epochs=50000 --mode="pq" --num_samples=100
python demo/sparsify_aquila_train.py --save_dir_name="interaction_aquila_reproduce" --gpu_id=1 --lr=1e-6 --epochs=50000 --mode="pq" --num_samples=100
python demo/sparsify_llama_train.py --save_dir_name="interaction_llama_reproduce" --gpu_id=1 --lr=1e-6 --epochs=50000 --mode="pq" --num_samples=100

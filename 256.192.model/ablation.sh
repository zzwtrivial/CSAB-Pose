echo "wo_light"
python ablation.py --model E_CPN50 --exp wo_light --active_light

echo "wo_denoise"
python ablation.py --model E_CPN50 --exp wo_denoise --active_denoise

echo "wo_fe"
python ablation.py --model E_CPN50 --exp wo_fe --active_fe
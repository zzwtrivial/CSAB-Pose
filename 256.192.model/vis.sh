#python uni_test.py --model E_CPN50 --path /home/wjm/MyFinalProject/pytorch-cpn/256.192.model/ablation_ckpt/wo_denoise-5.pth.tar --save wo_denoise
#python uni_test.py --model E_CPN50 --path /home/wjm/MyFinalProject/pytorch-cpn/256.192.model/ablation_ckpt/wo_fe-7.pth.tar --save wo_fe
#python uni_test.py --model E_CPN50 --path /home/wjm/MyFinalProject/pytorch-cpn/256.192.model/ablation_ckpt/wo_light-14.pth.tar --save wo_light

anno1="/home/wjm/MyFinalProject/pytorch-cpn/256.192.model/uni_test/wo_denoise-ll-Normal.json"
anno2="/home/wjm/MyFinalProject/pytorch-cpn/256.192.model/uni_test/wo_fe-ll-Normal.json"
anno3="/home/wjm/MyFinalProject/pytorch-cpn/256.192.model/uni_test/wo_light-ll-Normal.json"

result1="result_LL_normal/wo_denoise"
result2="result_LL_normal/wo_fe"
result3="result_LL_normal/wo_light"

mkdir $result1
mkdir $result2
mkdir $result3

python render.py --result $result1 --anno $anno1
python render.py --result $result2 --anno $anno2
python render.py --result $result3 --anno $anno3
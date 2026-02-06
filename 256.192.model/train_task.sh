# 冻结CPN训练（FeatEnhancer+CPN）
python uni_train.py --baseline checkpoint/archive/baseline-8.pth.tar --model FeatEn_CPN50 --exp feat-freeze --freeze

python uni_train.py --model E_CPN50 --exp chapter3
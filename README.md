# **mmseg_WS**


</br>

## 📰 **Contributors**

**CV-16조 💡 비전길잡이 💡**</br>NAVER Connect Foundation boostcamp AI Tech 4th

|민기|박민지|유영준|장지훈|최동혁|
|:----:|:----:|:----:|:---:|:---:|

</br>

## 📰 **Links**

- 비전 길잡이 Notion 📝( not finished )

## 📰 **Objective**
- weighted_sum head에 대한 성능 검증


## 📰 **Dataset**

||ADE20K|
|:----:|:---:|
||<img src="https://user-images.githubusercontent.com/113173095/217480114-cc9dbb9b-deaa-4620-bb78-920da8d06b84.png" width="300" height="150">|
|Purpose|Fine-tuning|
|Num_classes|150|
|Training set|20,210 images|
|Validation set|2,000 images|

```
|-- ADEChallengeData2016
    |-- image
    |   |-- train
    |   `-- val
    `-- mask
        |-- train
        `-- val
```
## WS Decode Head

<img src="https://user-images.githubusercontent.com/25689849/217508282-bb070e23-280f-4268-a2cc-2d7021c2eab7.svg">

- Stage Features Upsample
    - <img src="https://user-images.githubusercontent.com/103131249/217580836-fd09f784-b0b8-497b-b58f-7b64466106fd.png" width="200">
    
- **Weighted Sum 적용**
    - <img src="https://user-images.githubusercontent.com/103131249/217582409-9b7e3443-4b72-42c2-8017-2b539fbc8167.png" width="150">

---

## 📰 **Result**

## ⚙️ **Installation**

```bash
git clone https://github.com/revanZX/mmseg_WS.git
```

## 🧰 **How to Use**

### FineTuing Example

```bash
# with WS
python tools/train.py \
       ./configs_custom/pvt_v2/fpnws_pvt_v2_b5_ade20k_160k.py \
       --work-dir ../work_dir/pvtws_v2_b5 \
       --seed 1

# with NON WS
python tools/train.py \
       ./configs_custom/pvt_v2/fpn_pvt_v2_b5_ade20k_160k.py \
       --work-dir ../work_dir/pvt_v2_b5 \
       --seed 1
```

---
## 📰 **Directory Structure**

```
|-- 🗂 appendix          : 발표자료 및 WrapUpReport
|-- 🗂 segformer         : HuggingFace 기반 segformer 모델 코드
|-- 🗂 boostformer       : Segformer 경량화 모델 코드
|-- 🗂 imagenet_pretrain : Tiny-ImageNet encoder 학습시 사용한 코드
|-- 🗂 util              : tools 코드 모음
|-- Dockerfile
|-- train.py             : ADE20K Finetuning 코드
|-- eval.py              : 모델 Inference 결과 출력 코드
|-- requirements.txt
`-- README.md
```

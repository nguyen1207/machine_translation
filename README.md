# Vietnamese-English Machine Translation with VinaLLaMA-7B

This repository contains the code and data processing for finetuning VinaLLaMA-7B and VinaLLaMA-7B-chat in the paper "VinaLLaMA-7B: A Large-Scale Vietnamese-English Machine Translation Model" by [Hieu Pham](https://hieupham.com), [Dat Quoc Nguyen](https://www.datquocnguyen.com), [Thi Ngoc Diep Do](https://sites.google.com/view/thingocthanhdo), [Minh Nguyen](https://www.minhlab.site), and [Son N. Tran](https://sites.google.com/view/sontran) **on machine translation task**.

The model is finetuned on teencode and slang data from social media text data UIT-VSMEC (translated to English using GPT4), synthetic data (generated using GPT4), parallel dataset mt_eng_vietnamese (HuggingFace).

The instruction prompt used for finetuning is **MTInstruct, AlignInstruct, HintInstruct, ReviseInstruct** in the paper "Tuning LLMs with Contrastive Alignment Instructions for Machine
Translation in Unseen, Low-resource Languages" by Zhuoyuan Mao and Yen Yu.
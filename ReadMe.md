## 미디어젠 해커톤

​	딥러닝 기술 기반 감정 대화 분류 시스템 개발 (사용자 - 시스템간의 대화로부터 사용자의 감정상태 분류)





#### 주최/주관

------

​	한국정보화진흥원 / 미디어젠





#### 기간

------

​	2020.11.09 ~ 2020.11.20





#### 사용 모델

------

​	KoELECTRA





#### 환경

------

#### Data

- 주최측에서 제공된 데이터 (추후에 정보화진흥원에서 데이터를 공개할 예정)
  - 60개의 감정분류라벨
  - train data (json 형식, 15832개)
  - dev data (json 형식, 1717개)
  - test data (json 형식, 2073개)





#### GPU

​	GTX 1060 6GB





#### Package

```
torch >= 1.1.0
mxnet >= 1.4.0
gluonnlp >= 0.6.0
sentencepiece >= 0.1.6
onnxruntime >= 0.3.0
transformers >= 2.1.1, <= 3.0.2
attrdict==2.0.1
fastprogress==1.0.0
```





#### 결과

------

- Finetuning

  - 5epoch, 11500 step. (acc 기준 0.38 ~ 0.39 에서 상승 기미가 안보여서 중단)

- 기존 baseline (주관측 학습)
  
  - Albert로 학습
  - 대분류 50%, 소분류 30%


- 실제 제출 결과

  ![](https://github.com/siasky21/MZ_CEC_Hackathon/blob/main/img/result.JPG)
  - Baseline 보다 각각 11%, 12% 





#### References

------

[KoELECTRA](https://github.com/monologg/KoELECTRA)

[WellnessConversationAI](https://github.com/nawnoes/WellnessConversationAI)

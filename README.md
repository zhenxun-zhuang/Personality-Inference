# Personality-Inference

With every new encounter, one is assessed and another person’s impression of him/her is formed. The first impression could greatly affect or even decide how one will be treated afterwards; thus, it is important to figure out how to make a good first impression. Among all factors that can influence the impression one makes, personality plays the most important role. Although not directly observable, personality can be inferred from an individual’s expressions, gestures, and other external cues. Humans are very skilled and efficient in this that a quick glance is usually enough. Inspired by that, I try to enable machines to automatically generate personality inferences. By providing instant feedback of how one will be viewed by others, this work can help people present themselves better by changing their behavior in simple ways. In order to exploit information from both visual and audio channels just like humans do, studies are carried out along two lines. For the visual modality, a modified CNN is employed, while a logistic regressor is used in the audio part. These two parts are then ensembled to obtain final predictions. This work was based on the paper:

**[Deep Bimodal Regression for Apparent Personality Analysis](https://cs.nju.edu.cn/wujx/paper/eccvw16_APA.pdf)** 
Chen-Lin Zhang, Hao Zhang, Xiu-Shen Wei, Jianxin Wu

Also thanks to the developers of the **[MatConvNet](http://www.vlfeat.org/matconvnet/)** which I used in this project.

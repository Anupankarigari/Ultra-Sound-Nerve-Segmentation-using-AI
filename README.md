# Ultra-Sound-Nerve-Segmentation-using-AI

Ultra-Sound-Nerve-Segmentation-using-AI – Deep learning models for ultrasound nerve image segmentation, including U-Net, DenseNet, and SK-U-Net, with preprocessing, augmentation, and performance evaluation for medical image analysis.

# Abstract
This work is aimed at the performance of six deep learning models on an image segmentation task using the dataset provided by Kaggle in the competition "Ultrasound Nerve Segmentation". The data includes 11,270 images of ultrasound in gray scale and their respective binary masks, split 70% for training and 30% for testing. Pre-processing steps included normalizing and resizing the data and several augmentation techniques to make the model robust and increase its generalization ability. Among them, DenseNet performed the best segmentation with a Dice Coefficient of 0.3067 and IoU of 0.1811, benefiting from the dense connectivity that promotes feature reuse and gradient flow. However, U-Net, while having a high validation accuracy of 0.9216, was not very good at fine-grained segmentation. MobileNet and TransCGUNet tended to overfit and, even with high overall accuracy, the precision of their segmentation was poor.

The SK-U-Net system uses Selective Kernel modules to dynamically adjust the size of the receptive field for substantial improvement in segmentation capability over U-Net. Thus, in a Dice Coefficient of 0.2376 and IoU of 0.1348, its validation accuracy is high (0.9845), but low losses reflect that it generalizes well to more complex datasets. Accordingly, this work emphasizes the design of architectures tailored to specific segmentation challenges. With a novel approach in SK-U-Net for medical image analysis, its architecture strikes an excellent trade-off between precision in segmentations and generalization capability. Thus, refinements of SK-U-Net with consideration of its diverse applications could achieve improved diagnostic performance and, correspondingly, assist with clinical treatment options.

# Keywords: Artificial Intelligence; Deep learning; Convolutional Neural Networks; Transfer Learning.

## 1.	Introduction:
Ultrasound-guided procedures have become indispensable in clinical practice, especially within regional anesthesia, where identifying peripheral nerves is crucial to performing safe and effective nerve blocks. Indeed, an accurate nerve location can reduce complications such as nerve injury, hematoma, and toxicity of the anesthesia injected, hence improving the recovery rate. (Gungor et al., 2021). However, several challenges are related to manually segmenting nerves in ultrasound images. These challenges include variability in nerve morphology, the inherently low contrast between nerves and surrounding tissues, and overall ultrasound image noise that hinders the practitioner from consistently identifying and delineating nerves with precision. (Wu et al., 2021) State that there are several challenges associated with identifying nerves using ultrasound imaging, including nerve morphologic variability, poor contrast between nerves and the surrounding tissues, and overall ultrasound image noise, all of which affect a practitioner's ability to identify and delineate nerves accurately.

Conventionally, ultrasound nerve segmentation has relied on manual annotation by expert clinicians, a process that is invariably very time-consuming, labor-intensive, and prone to interobserver variability. (Festen et al., 2021). This has, therefore, led to an increased interest in the development of automated segmentation techniques that can assist clinicians in performing more accurate and efficient interventions under ultrasound guidance. In particular, deep learning models, and specifically CNNs, are newer AI techniques that have shown a lot of promise recently in overcoming these challenges by automatically providing accurate segmentation of nerves.

Works have demonstrated phenomenal success regarding the application of deep learning models in the segmentation of nerves from ultrasound images. (A. Huang et al., 2022) Proposed an Attention-VGG16-UNet to couple CNNs with attention mechanisms in segmenting the median nerve within ultrasound images. This model further improved the accuracy of segmentation by making the model concentrate on the most informative parts of the image and also reduced interference caused by noise and low contrast. Similarly, (Festen et al., 2021)Used the U-Net architecture in segmenting the median nerve in carpal tunnel syndrome cases with highly accurate results and increased diagnostic confidence.

On the other hand, (Ding et al., 2020)Developed BPMSegNet, a multiple-instance segmentation model tailored for segmenting the brachial plexus in ultrasound images. This approach not only improved precision in segmentation but also showed scalability on large datasets of complex anatomic structures. In this way, deep learning methods allowed the development of models able to learn more consistently and accurately the spatial and morphological features of nerves compared to manual segmentation methods, such as that by (Ding et al., 2020)

However, most of the current models target 2D ultrasound images, and as such, may be seriously limited due to the loss of important spatial and morphological details. Because of these limitations, the trend has been to employ region-aware global context modeling to obtain more holistic nerve features. Discussion can be found in the work by (Wu et al., 2021). In this manner, for instance, it will be possible for the model to achieve an improved distinction between nerves from surrounding tissues by modeling global features of an image together with local context and thus improve segmentation results.

There is a very fast-growing area concerning the combination of real-time clinical applications with AI-based models, which already guarantees extremely great potential benefits in patient care. For example, (Gungor et al., 2021) Had an accuracy study showing the feasibility of real-time AI anatomy identification during an ultrasound-guided peripheral nerve block procedure. The prospects of automating nerve identification in real-time, with the advent of AI, and reducing dependence on manual approaches to enhance efficiency in ultrasound-guided interventions remain with these models.

Moreover, there are temporal dependencies embedded in higher AI models like DeepNerve, which assure consistency in nerve segmentation over successive frames of ultrasound images. This is quite useful in dynamic structures, such as the median nerve. (Horng et al., 2020). These advances not only raise diagnostic precision but also ease the workload on the clinician by allowing him to focus efforts on clinical decision-making rather than manual interpretation of images.

Besides improving segmentation accuracy, some AI models, such as RepVGG (X. Ding et al., 2021), were designed to preserve the computational efficiency characteristic of deep learning models. By fine-tuning CNN architectures toward performance and efficiency, these models have the potential to be implemented into real-time clinical workflows without invoking significant additional computational resources.

Despite such promising developments, challenges remain. However, variability in ultrasound imaging quality, anatomy variability among patients, and sometimes the presence of artifacts may further challenge AI models. These are very promising; however, studies to fine-tune these models are still necessary to establish the effectiveness of such systems in various clinical scenarios among disparate patient populations. (Baby & Jereesh, 2017). Translational work will certainly be required to bring AI-driven nerve segmentation models into the clinical realm, where different groups of researchers, clinicians, and software developers collaborate to ensure usability and integration within the context of existing ultrasound machines.

This work aims to contribute to this increasingly developing body of research by proposing an AI-based model that integrates deep learning techniques with region-aware global context modeling for the improved segmentation of nerves in ultrasound images. The final goal will be a model that can be applied in real-time clinical scenarios with very high accuracy of segmentation, computational cost reduction, and noise interference abatement to enhance the safety and efficiency of procedures guided by ultrasound.


**2.	Background/Literature Review**
Hadjerci et al., (Hadjerci et al., 2015) The authors introduce a novel feature selection algorithm in this paper, the nerve localization framework on UGRA. Since the approach relies on statistical and learning models, automation of the nerves detection when using ultrasound guidance is a challenging task due to some issues related to noise and artifacts that would be very useful to the anesthetist. These results mirror that the proposed technique is effective and efficient in detecting the nerve zone while performing considerably well from existing techniques presented in the literature, with an accuracy of 82% on one dataset and 61% on another untrained dataset.

Ronneberger et al., (Ronneberger et al., 2015) The authors present a new architecture for deep learning-based segmentation of biomedical images. The authors introduce a CNN called U-Net that is particularly designed to resolve two challenging problems in segmenting biomedical images: the availability of few training data and high-accuracy boundary localization. The U-Net architecture comprises an extracting path, or contracting path, and an expansive path, each linked by skip connections. The effectiveness of U-Net on a variety of datasets, including electron microscopic images of neural structures and histological sections of the kidney, was performed by the authors. Results have been presented that show that U-Net achieved a high degree of accuracy with state-of-the-art methods, hence becoming one of the power tools for biomedical image segmentation tasks.

Hadjerci et al., (Hadjerci, Hafiane, Conte, et al., 2016) In this paper, the challenge is presented for detecting and segmenting nerves in ultrasound images for UGRA. An efficient approach has been proposed to detect and segment nerves, along with a review and evaluation of the performance of existing methods in the literature. The overall architecture of the proposed system has been segregated into four main stages: de-speckling filter, feature extraction, feature selection, and classification and segmentation. The authors performed a comparative study on each stage to measure the impact on the overall system. Using sonographic videos from 19 volunteer patients, they assessed the effect of training set size and evaluated consistency through a cross-validation technique. The proposed framework achieved high scores, with averages of 80% of 1900 tested images, hence proving valid and useful for UGRA applications.

Hadjerci, Hafiane, Morette, et al., (Hadjerci, Hafiane, Morette, et al., 2016) The authors in this work present the first fully automatic system for detecting the regions of interest and generating needle trajectories in UGRA. It addresses two important steps of UGRA: anatomical structure recognition and needle steering to the target region. The proposed system is two-staged. The automatic localization and segmentation of nerves and arteries in ultrasound images by the machine learning algorithm classifying into multimodels using an active contour will be done first. Based on the outcome from step one, a path planning algorithm will be developed to obtain an optimal needle insertion trajectory. Experiments on the individual modules of the detection framework were performed to demonstrate the performance of the proposed system, and the overall framework was compared against the existing methods. For assessing the robustness, the proposed method was tested using two datasets each acquired at different times. The proposed assistive system, robust and feasible in the practice of UGRA, has been confirmed by experiments. The method also points out a way to enhance safety and to generalize the system within medical facilities when expertise from practitioners is scarce.

Baby & Jereesh (Baby & Jereesh, 2017)In this conference paper, the authors will propose a method to segment US images of the BP nerve bundle. Their approach first de-speckles the training set (n=5640) to reduce background speckle noise and then onward to training using the popular U-Net architecture. Training was also carried out on a traditional SVM to compare performances. On 5508 test images, the average Dice scores for the U-Net and SVM were 0.71 and 0.64 respectively.

Smistad et al.,  (Smistad et al., 2017) The authors in this article proposed a system that would facilitate the painful processes of Ultrasound-guided femoral nerve block to less experienced physicians. It typically guides the user in moving the ultrasound probe to investigate the region of interest and reach the target site for needle insertion. It automatically segments in real-time the femoral artery, the femoral nerve, the fascia lata, and the fascia iliaca to aid in interpreting the 2D ultrasound images and surrounding anatomy in 3D. The system was evaluated on 24 ultrasound acquisitions from six subjects and the results were compared with those of an expert anaesthesiologist. The mean target distance had an average of 8.5 mm with a standard deviation of 2.5 mm, and the average absolute differences of the segmentations of the femoral nerve and fascia were around 1-3 mm.

Zhao & Sun (H. Zhao & Sun, 2017) The authors propose a computer vision-based method for automatic segmentation of medical images using CNNs in this paper. The proposed network architecture is a modification of the U-Net model with the use of inception modules instead of regular convolutional layers, jointly with batch normalization technique which reduces parameters and speeds up training without loss of accuracy. Also, the authors replace the binary cross entropy loss function with the Dice coefficient. The proposed model averaged a Dice score of 0.653 at a model size of 5M parameters, while the U-Net scored an average of 0.658 with 31M parameters.

Liu et al.,  (Liu et al., 2018) The authors of this work have designed a deep adversarial neural network to deal with challenges related to segmenting the BP nerve on US images. The authors developed a segmentation network based on the VGG network variant. Furthermore, the authors introduced a discriminator network that would impose the anatomical dependencies assessing the quality of segmentation. Finally, elastic deformations are introduced to the dataset to simulate anatomic variability across a range of patient profiles. A mean intersection over union (mIOU) score of 73.29% was obtained for the author's model incorporating deep adversarial networks.

Smistad et al., (Smistad et al., 2018) Segmenting nerves and blood vessels for ultrasound-guided axillary nerve block procedures using neural networks The authors here employ a deep convolutional neural network to outline the main structures including nerves and blood vessels on the ultrasound images collected during the axillary nerve block procedures. A dataset of 49 subjects is collected and used for training and testing of the neural network. The authors review various augmentations of images, rotation, elastic deformation, shadows, and horizontal flipping. The authors have done cross-validation to assess the neural network and find that with regards to blood vessels, these were most easily detected with precision and recall above 0.8. The median and ulnar nerves give the highest F-scores among detected nerves, with scores of 0.73 and 0.62 respectively, while the radial nerve was the most challenging nerve to detect with an F-score of 0.39. The authors state that image augmentations can improve the F-score by up to 0.13, while the combination of all augmentations gives the best results. They do, however, realize that the obtained precision and recall values are not yet good enough and estimate detecting deep nerves in ultrasound may require the support of more data while concerning temporal and anatomical models could help achieve better accuracy.

Alkhatib et al., (Alkhatib et al., 2019)In the current study, the authors compare thirteen recent deep-learning trackers on various types of nerves concerning their accuracy, consistency, time complexity, and ability to adapt to several different situations that a nerve may be found in, such as loss of shape information or disappearance of tissue. These trackers are tested on the median and sciatic nerve US dataset that consists of 10,337 still images from 42 adult patients. These results thereby prove that deep learning trackers of this genre yield highly satisfactory performances for different kinds of nerves, and therefore confirm their potential in UGRA procedures.

Feng-Ping & Zhi-Wen (Feng-Ping & Zhi-Wen, 2019) The authors of this work initiate their discussion by pinpointing the major lacunae of some traditional methods of image segmentation, as well as those involving standard CNNs within the medical image segmentation paradigm. They put forward a new algorithm drawing inspiration from the feedback mechanism of the human visual cortex. They further propose two new algorithms for solving the optimization problem for feedback, which are based on the greedy strategy. They then present a medical image segmentation algorithm that exploits this feedback mechanism in the interior of the CNN. In the process, deep features of the images are learned and extracted by training the convolutional neural network models with unlabeled image block samples and then used for classifying pixel block samples in the medical image to be segmented. Finally, the threshold segmentation and morphological approaches will be further optimized based on this model. The new method has a high overall segmentation accuracy and adaptability for different medical images.

Huang et al., (C. Huang et al., 2019) The authors in this work proposed a method for identifying the femoral nerve block area using ultrasound images. The system they proposed targeted mainly the less experienced operators. They developed a dataset of ultrasound images depicting the femoral nerve block and labeled them accordingly. They used the U-net framework to train their model by segmenting the region of interest in the images. The IoU and accuracy metrics were used for model performance. Thus, median IoU results are 0.722, 0.653, and 0.644 for the training set, the development set, and the test set, respectively. The accuracy in segmenting test set images was 83.9%. Besides, the authors have obtained by 10-fold cross-validation a median IoU of 0.656 and accuracy within the range from 82.1% to 90.7%. In conclusion, the authors said that a U-net-trained model showed acceptable performance regarding segmentation in the femoral-nerve region and can be applied to clinical practice potentially.

Y. Weng et al., (Y. Weng et al., 2019) The success of the U-net and its variants inspires the authors to extend neural architecture search into medical image segmentation. They present the procedure of searching in three types of primitive operation sets for the search space to get two cell architectures, namely DownSC and UpSC. These are utilized in NAS-Unet, which is a U-shaped network for semantic segmentation. In this process, DownSC and UpSC architectures will be updated simultaneously while incorporating the differential architecture strategy. The authors tested their approach with Promise12, Chaos, and ultrasound nerve datasets obtained using MRI, CT, and Ultrasound, respectively. Their NAS-Unet was trained on PASCAL VOC2012 and when tested on the same referred medical image datasets as the one in this work, showed a better performance using a much lower number of parameters when compared with the U-net and one of its variants.

Singarayan et al.,  (Singarayan et al., 2020) This work proposes an automated pipeline using a Mask R-CNN approach coupled with U-net for the detection and segmentation of the IAC and its respective nerves. Mask R-CNN uses the RESNET50 model as the backbone to localize the IAC, while U-net learns the features to segment the IAC and nerves. Testing of the proposed method is performed on clinical datasets of 50 patients consisting of adults and children. Evaluation metrics used include IoU for IAC localization and the Dice similarity coefficient for segmentation. The performance of the methodology was impressively good, where the RESNET50 and RESNET101 reached a mean IoU of 0.79 and 0.74, respectively. In terms of segmentation, the proposed method outperformed region growing and PSO methods with a Dice similarity coefficient of 96%. These findings indicate that the proposed AI system can help support radiologists by allowing the accurate localization and segmentation of the IAC and its nerves.

Bowness et al., (Bowness et al., 2021) In this article, the authors describe an AI system's contribution in helping to identify anatomical structures with ultrasound guidance during regional anesthesia. Since the common difficulties of the anesthesiologists usually imply a possible benefit of this AI system, the regional anesthesia experts reviewed 40 ultrasound scans, comparing unmodified and AI-highlighted videos side-by-side. They assessed the overall performance of the AI system in highlighting the nerves and vessels, the utility of the highlighting in identifying specific structures, and how well the highlighting aids in confirming correct ultrasound views for less-experienced practitioners. It provided specific anatomical structure identification in 99.7% of cases and confirmed the correctness of ultrasound views in 99.3% of scans. Although these results should be further investigated, they illustrate AI technology's possible role in enhancing clinical practice and renewing interest in the field of clinical anatomy.

Cho et al., (Cho et al., 2021) It proposes a novel spatially adaptive weighting scheme for medical image segmentation to improve the performance of U-Net-based architectures. Various convolutional frameworks such as VGG, ResNet, and Bottleneck ResNet structures are used in the scheme, replacing the up-convolutional layer with a bilinear up-sampling method. Performance comparisons conducted on three different medical imaging datasets achieved very significant improvements. The best IoU and Dice scores among the methods tested in this paper have been obtained for the network based on the ResNet framework with the proposed self-spatial adaptive weighting block. Concretely, using the nerve dataset, the IoU increased by 3.01% and Dice by 2.89% with the proposed block incorporated with the ResNet framework, hence showing very good potential for the new approach regarding image segmentation tasks.

Gungor et al., (Gungor et al., 2021) The precision of a real-time AI anatomy identification tool developed to aid in interpreting UGRA images was studied. With the software, students of anesthesiology performed different types of blocks on 40 healthy volunteers (20 female and 20 male). After the software had confirmed that anatomical landmarks for each block had been scanned 100% successfully, the ultrasound images were saved and subsequently analyzed by expert validators. When the trainees reached 100% scan success, the accuracy ratings provided by the validators became consistent. Save for TAP blocks, which showed an inverse relationship to both age and BMI, the scores did not vary significantly according to participant demographics. These findings suggest that AI can support anesthesiologists in the conduction of UGPNB with correct anatomical structure identification during real-time sonography.

Marzola et al., (Marzola et al., 2021) The authors introduce a deep learning approach for segmenting cross-sectional areas (CSA) in transversus musculoskeletal US images while also providing a quantitative grayscale analysis of the images. This dataset has 3917 images coming from 1283 subjects with expert annotations. Bland-Altman plots, grayscale analysis, and correlation analysis are applied in comparing ground truth predictions against automatic segmentation predictions of experts in the test set. Results on the test set, when compared to manual annotation by human experts, attain a precision of 0.88 ± 0.12 and recall of 0.92 ± 0.09 but achieved somewhat lower values for abnormal muscles. Besides, intra-class and Pearson's correlation coefficients also reflected excellent agreement in the analysis. The CSA segmentation model was well performed and provided the manual operator with information on z-score grayscale.

D. Tian et al., (D. Tian et al., 2022) The review is undertaken to compare twelve deep learning models performances while segmenting the brachial plexus. The dataset was composed of 340 images annotated by three different annotators, all of whom were anesthesiologists. Of the twelve different models compared, the U-Net architecture had the highest mean IoU of 68.5% but is only able to process 15 frames per second due to the size of the model. The LinkNet architecture came next best in performance with a mean IoU8 score of 66.27%, able to process 142 frames per second.

Almasi et al.,  (Almasi et al., 2021) The author of this article purports that there lacks a robust tool to assess US-guided nerve block quality. Therefore, the study describes an approach to assess US-guided inter scalene-supraclavicular blocks and axillary-supraclavicular blocks on 93 patients. Sensory, motor, coping, and postoperative pain, SMCP metric, were documented for each patient, and the quality of the anesthesia graded by an anesthesiologist assessed (QAGA). Also, results of no significant difference in QAGA for both block groups were demonstrated. What's more, 97.8% of patients were in the Excellent and Good categories with SMCP, whereas 86% were with QAGA (p<0.001).


**3.	Materials and Methods**

For this project, the data is split into training and testing subsets-70% of images and masks for training and 30% for testing. This manual split ensures that the dataset is comprehensive and well-balanced for model development and evaluation. Preprocessing included normalization of pixel intensity values in the range of [0, 1], resizing images and masks to have the same input dimensions, and augmenting data by rotation, flipping, and cropping to increase the performance and robustness of the model. This dataset would represent a high-quality resource for the training of machine learning models to automate nerve structure segmentation in medical imaging with great potential to improve diagnostic precision and help surgical planning.

**3.1	Data Description**

The dataset used in this project was obtained from the Kaggle competition "Ultrasound Nerve Segmentation." This was a competition designed to segment nerve structures in ultrasound images of the neck. It contains 11,270 grayscale ultrasound images that are kept in a folder, with corresponding binary segmentation masks kept in another folder. Each mask is structured so that the nerve structures are in white (1), and the background is black (0). The file naming convention follows a clear formatting, for example, 10_100_mask.tif, corresponding to their photo counterparts, so they can be easily paired. All files were provided in.tif format to have high-quality data for a segmentation task. 

**3.2 Deep Learning Techniques**

The project applies machine learning methods. These models were selected due to their ability to model nonlinear relationships in the data and capture complex trends efficiently.

**3.2.1 U-Net**

U-Net is a CNN architecture mainly used for image segmentation. It has a symmetric encoder-decoder structure: the encoder captures the spatial features by successive convolution and pooling, while the decoder reconstructs the image by transposed convolutions. The "U" shape is accomplished by skip connections that directly pass feature maps from the encoder to the decoder, preserving the spatial context and fine-grained details. The output typically comprises of a pixel-wise classification map.(Horng et al., 2020)


<img width="940" height="470" alt="image" src="https://github.com/user-attachments/assets/a950dc0a-912f-46ec-a5db-87b7bf8f6bab" />

 
**3.2.2 DenseNet**

DenseNet, short for Dense Convolutional Network, increases the flow of information by directly connecting every layer to all other layers in a block. It guarantees feature reuse, improves the flow of gradients, and reduces parameters. The idea is that each layer uses the feature maps of all the previous layers as input and its output is used by all subsequent layers. A DenseNet model is composed of dense blocks and transition layers.(Ma et al., 2024)

<img width="990" height="326" alt="image" src="https://github.com/user-attachments/assets/9f3885ae-672c-4d97-b526-1ae1dae1afb2" />

 
**3.2.3 VGG**

 VGG is a deep convolutional neural network that is basically simple and uniform in architecture. It relies on small 3x3 convolution filters with depth that is increased by adding more layers. The architecture has emphasized uniformity: gradually reducing spatial dimensions through max-pooling layers while doubling the number of filters. The VGG models, such as the VGG16 and VGG19, represent the class of models for image classification.(A. Huang et al., 2022)

 <img width="940" height="413" alt="image" src="https://github.com/user-attachments/assets/503d59da-5c6f-47e7-b565-fc429c57260b" />

 
**3.2.4 MobileNet**
MobileNet is a lightweight deep learning model for mobile and embedded vision applications. It uses depthwise separable convolutions as the main building block to reduce computations and the number of parameters required. The network diminishes each convolution into two steps: depthwise convolution for spatial filtering and pointwise convolution (1x1) to combine features.(Pu et al., 2022)

<img width="940" height="415" alt="image" src="https://github.com/user-attachments/assets/ece02b3d-4cb9-4ebb-82da-bfb45b73eae6" />

 
**3.2.5  TransCGUNet**

TransCGUNet incorporates transformer-based architectures with Convolutional Gated U-Net for segmentation tasks. While transformers handle global context, the CGUNet manages spatial and local features of an image. This hybrid model serves as a perfect vehicle in the modeling of both global dependencies and fine details, particularly in medical image segmentation.(Gujarati et al., 2023)

<img width="940" height="443" alt="image" src="https://github.com/user-attachments/assets/665d2adb-d920-4a61-b36a-02d1fb36fba4" />

 
**3.2.6 SK-U-NET (MY MODEL)**

SKUNet is a variant of the U-Net architecture that integrates attention mechanisms in its pathways to selectively choose relevant features and reduce noise in image segmentation tasks. Selective kernel units dynamically adjust their receptive field of view on the most relevant features at different scales. It proves especially effective against complex and noisy datasets.(Ansari et al., 2024)

<img width="940" height="406" alt="image" src="https://github.com/user-attachments/assets/285435d4-d216-4630-abf7-4223e91416d4" />

**4.	Results and Discussion**

**4.1 Performance Evaluation Measures**

**4.1.1 Dice Coefficient (Dice Similarity Index)**

The Dice Coefficient can be understood as the metric to estimate similarity between the predicted and ground truth binary masks for a segmentation task. It estimates how much the two masks overlap with each other. Its range goes from 0 to 1, where 1 stands for a perfect overlap and 0 means there is no overlap at all. Values less than 1 indicate partial overlap, and the lower the value, the poorer the accuracy of segmentation. This metric is very useful for tasks in medical image analysis, among others, that require accurate location of boundaries because correct segmentations are related to diagnosis and treatment planning.(Jagtap et al., 2023)

**Formula:**

<img width="328" height="104" alt="image" src="https://github.com/user-attachments/assets/85c395b3-07f0-4637-85eb-52eb9b5af213" />


 
**4.1.2 Intersection over Union (IoU)**

The IoU, also known as the Jaccard Index, is one of the standard metrics for segmentation accuracy, showing the ratio between the intersection-usually understood as the area that predicted and true binary masks share-and their union, understood as the area of their combination. It goes from 0 to 1, where 1 reflects perfect overlap; that is, perfect alignment between the predicted mask and the mask used as ground truth. Conversely, 0 denotes no overlap; there is complete mismatch. The values range from 0 to 1 and reflect the degree of accuracy, where a higher value indicates good performance in segmentation. IoU finds broad applications in autonomous driving, medical imaging, and other uses in computer vision, where accurate boundary detection is crucial. IoU provides quantitative information on both the agreement and disagreement between masks, hence providing great insight into model performance that enables focused improvements.(C. Huang et al., 2019)

**Formula:**

<img width="384" height="185" alt="image" src="https://github.com/user-attachments/assets/61b1d6f5-7132-426d-a9f0-62f16f2bafd1" />

 
**4.1.3 Sensitivity**

Sensitivity, also referred to as recall, is the ratio of actual positive areas, say nerves, which were correctly predicted by a model. This is a measure of the model's ability to find true positives ranging in value from 0 to 1. The ideal sensitivity is 1.0, meaning perfect recall without false negatives or missed regions. Values less than 1 indicate the presence of false negatives, and the more minimal the value, the greater the number of missed areas. Sensitivity is very important in applications such as medical imaging, where missing a critical region may be dangerous. With high sensitivity, the model will capture most, if not all, of the positive regions, reducing the risk of missing important features.(Wu et al., 2021)

**Formula:**

<img width="940" height="169" alt="image" src="https://github.com/user-attachments/assets/1d795a26-0292-43d4-81e0-ba5580920650" />

 
**4.1.4 F1 Score**

The F1 Score is the harmonic mean of precision and sensitivity, thus giving a measure that balances the trade-off between false positives and false negatives in one convenient number. It takes values between 0 and 1, with an ideal value of 1.0 indicating a perfect balance of precision and recall, where the model gives high accuracy on true positives while keeping both false positives and false negatives low. Values less than 1 indicate some kind of trade-off, with smaller values indicating poor overall performance of both precision and recall. The F1 Score finds applications in cases where a good balance between these two measures needs to be ensured, in the case of imbalanced datasets, where either optimizations for precision or recall could give misleading evaluations on their own.(Wang et al., 2023)

**Formula:**

<img width="724" height="165" alt="image" src="https://github.com/user-attachments/assets/365651de-d6e3-40b2-ada9-982f95d9c84e" />


**4.2 Experimental Results**


<img width="940" height="300" alt="image" src="https://github.com/user-attachments/assets/69e5e160-a2d0-4527-836c-9996f66865ae" />

 
It shows the result comparison table of six deep learning models, namely U-Net, DenseNet, VGG, MobileNet, SKUNet, and TransCGUNet, on an image segmentation task, using Dice Coefficient, IoU, Sensitivity, F1 Score, accuracy, and loss as metrics. Each model shows different strengths and weaknesses, giving insight into their suitability for specific segmentation challenges. U-Net, though simple, achieves a very high validation accuracy of 0.9216 but falters on the segmentation precision as reflected by its low Dice Coefficient value of 0.1737 and IoU of 0.0951. These indicate that though it learned the overall data distribution very well, its basic architecture lacks the sophistication to capture fine-grained details, which is very much needed for complex segmentations.

DenseNet was outstanding in segmentation, though, with a Dice Coefficient of 0.3067 and IoU of 0.1811. Because of the rich and dense connectivity of the layers in it, the reusing of features and flow of gradients are enhanced, allowing the high training accuracy of 0.9798 and a validation accuracy of 0.9725. Therefore, DenseNet is the strongest regarding the accuracy of segmentation. VGG provides a balance between segmentation and classification, with a Dice Coefficient of 0.2668 and IoU of 0.1539. While it has maintained the highest validation accuracy of 0.9810, its relatively high validation loss of 0.7388 indicates that it could be further improved by mechanisms that adapt better to changes in features for fine detail segmentation.

Although MobileNet is light and efficient, it could not perform well for segmentation with very low values for both Dice Coefficient and IoU, 0.0011 and 0.0005 respectively. Surprisingly, the model has a high training and validation accuracy of 0.9897 and 0.9865, respectively, which indicates overfitting when high-performance models do not match the poor segmentation precision. The network includes features of transformers and U-Net with gates, while the segmentation performance of TransCGUNet is low-0.0120 for Dice Coefficient and 0.0060 for IoU, probably due to overfitting or unsatisfactory adaptation of transformer methods to the particular segmentation task. On the other hand, the network has a very high accuracy of training, 0.9721, and a validation accuracy of 0.9750, showing promise when fine-tuned further.

**SK-U-Net_New Model:**

In comparison, my new model, SK-U-Net, which integrates the Selective Kernel modules into the U-Net model, significantly enhances the segmentation capability of U-Net. Achieving a Dice Coefficient of 0.2376 and Intersection over Union of 0.1348, SK-U-Net exhibits high validation accuracy of 0.9845 and low loss values of 0.0235 for training and 0.0542 for validation, hence generalizing very well on complex datasets. Among these models, SK-U-Net is a promising new approach that provides a good balance between the precision of segmentation and generalization capability. Its novel architecture with embedded Selective Kernel modules for dynamic receptive field adjustment will make it a strong contender to handle complex and noisy datasets and mark an important advancement in research related to image segmentation.

**Note:** SKUNet is a novel model that integrates Selective Kernel modules to dynamically adjust receptive fields, offering improved segmentation performance in challenging tasks.

**5.	Conclusion:**

This study finally gives an elaborative comparison between six deep learning models: U-Net, DenseNet, VGG, MobileNet, TransCGUNet, and the proposed SK-U-Net for image segmentation. Each of the models exhibited strengths and weaknesses that were clearly different with respect to Dice Coefficient, IoU, sensitivity, F1 Score, accuracy, and loss. Despite the high validation accuracy of 0.9216, U-Net showed poor fine-grained segmentation precision with a low Dice Coefficient of 0.1737 and IoU of 0.0951. This can be explained by its basic architecture that is not too powerful for such complex challenges in segmentation. On the other hand, DenseNet had the best performance with respect to segmentation accuracy: a Dice Coefficient of 0.3067 and IoU of 0.1811 due to its dense connectivity among layers that promotes feature reuse and gradient flow.
VGG showed a good balance between segmentation and classification performance with a Dice Coefficient of 0.2668, IoU of 0.1539, and the highest validation accuracy of 0.9810. Still, its comparably high validation loss indicated that it might be further tuned for fine-detail segmentations. MobileNet, while very efficient and light, demonstrated poor segmentation precision with extremely low values of Dice Coefficient-0.0011 and IoU-0.0005, likely because of overfitting. TransCGUNet, a model that incorporates transformers with U-Net-like gating mechanisms, performed similarly poorly, reaching a Dice Coefficient of 0.0120 and IoU of 0.0060 in segmentation. Both these models failed to generalize well enough for accurate segmentation despite high accuracies in their training and validation.

The proposed SK-U-Net achieved a much better performance than U-Net with a Dice Coefficient of 0.2376 and IoU of 0.1348, along with very high accuracy in validation (0.9845) and low loss values. By incorporating Selective Kernel modules, the SK-U-Net adaptively adjusts its receptive fields to handle such complex and noisy datasets. Such a design provides a new balance between segmentation precision and generalization capability, hence this SK-U-Net might be an advance in image segmentation research. The future work can be the fine-tuning of SK-U-Net, the extension to a broader segmentation task, and can point out the promising prospects it has for more challenging datasets, having a high accuracy and robustness level.

**6.	Kaggle Link**

https://www.kaggle.com/code/anupaankarigari/ultrasound-nerve-segmentation-using-ai


**References:**

Alkhatib, M., Hafiane, A., Vieyres, P., & Delbos, A. (2019). Deep visual nerve tracking in ultrasound images. Computerized Medical Imaging and Graphics, 76, 101639. https://doi.org/10.1016/j.compmedimag.2019.05.007

Almasi, R., Rezman, B., Kovacs, E., Patczai, B., Wiegand, N., & Bogar, L. (2021). New composite scale for evaluating peripheral nerve block quality in upper limb orthopaedics surgery. Made in Hungary: Scientific and Skill Proposals from Hungarian Trauma Surgeons, 52, S78–S82. https://doi.org/10.1016/j.injury.2020.02.048

Ansari, M. Y., Mangalote, I. A. C., Meher, P. K., Aboumarzouk, O., Al-Ansari, A., Halabi, O., & Dakua, S. P. (2024). Advancements in Deep Learning for B-Mode Ultrasound Segmentation: A Comprehensive Review. IEEE Transactions on Emerging Topics in Computational Intelligence.

Baby, M., & Jereesh, A. S. (2017). Automatic nerve segmentation of ultrasound images. 2017 International Conference of Electronics, Communication and Aerospace Technology (ICECA), 107–112. https://doi.org/10.1109/ICECA.2017.8203654

Bowness, J., Varsou, O., Turbitt, L., & Laurent, D. (2021). Identifying anatomical structures on ultrasound: Assistive artificial intelligence in ultrasound‐guided regional anesthesia. Clinical Anatomy, 34. https://doi.org/10.1002/ca.23742

Cho, C., Lee, Y., Park, J., & Lee, S. (2021). A Self-Spatial Adaptive Weighting Based U-Net for Image Segmentation. Electronics, 10, 348. https://doi.org/10.3390/electronics10030348

D. Tian, B. Zhu, J. Wang, L. Kong, B. Gao, Y. Wang, D. Xu, R. Zhang, & Y. Yao. (2022). Brachial Plexus Nerve Trunk Recognition From Ultrasound Images: A Comparative Study of Deep Learning Models. IEEE Access, 10, 82003–82014. https://doi.org/10.1109/ACCESS.2022.3196356

Ding, Y., Yang, Q., Wu, G., Zhang, J., & Qin, Z. (2020). Multiple Instance Segmentation in Brachial Plexus Ultrasound Image Using BPMSegNet. ArXiv, abs/2012.12012. https://api.semanticscholar.org/CorpusID:229348944

Feng-Ping, A., & Zhi-Wen, L. (2019). Medical image segmentation algorithm based on feedback mechanism convolutional neural network. Biomedical Signal Processing and Control, 53, 101589. https://doi.org/10.1016/j.bspc.2019.101589

Festen, R. T., Schrier, V. J. M. M., & Amadio, P. C. (2021). Automated Segmentation of the Median Nerve in the Carpal Tunnel using U-Net. Ultrasound in Medicine & Biology, 47(7), 1964–1969. https://doi.org/10.1016/j.ultrasmedbio.2021.03.018

Gujarati, K. R., Bathala, L., Venkatesh, V., Mathew, R. S., & Yalavarthy, P. K. (2023). Transformer-based automated segmentation of the median nerve in ultrasound videos of wrist-to-elbow region. IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control.

Gungor, I., Gunaydin, B., Oktar, S. O., M.Buyukgebiz, B., Bagcaz, S., Ozdemir, M. G., & Inan, G. (2021). A real-time anatomy ıdentification via tool based on artificial ıntelligence for ultrasound-guided peripheral nerve block procedures: An accuracy study. Journal of Anesthesia, 35(4), 591–594. https://doi.org/10.1007/s00540-021-02947-3

Hadjerci, O., Hafiane, A., Conte, D., Makris, P., Vieyres, P., & Delbos, A. (2015). Nerve Localization by Machine Learning Framework with New Feature Selection Algorithm (Vol. 9279). https://doi.org/10.1007/978-3-319-23231-7_23

Hadjerci, O., Hafiane, A., Conte, D., Makris, P., Vieyres, P., & Delbos, A. (2016). Computer-Aided Detection system for nerve identification using ultrasound images: A comparative study. Journal of Informatics in Medicine Unlocked 2352-9148, 3. https://doi.org/10.1016/j.imu.2016.06.003

Hadjerci, O., Hafiane, A., Morette, N., Novales, C., Vieyres, P., & Delbos, A. (2016). Assistive system based on nerve detection and needle navigation in ultrasound images for regional anesthesia. Expert Systems with Applications, 61, 64–77. https://doi.org/10.1016/j.eswa.2016.05.002

Horng, M.-H., Yang, C.-W., Sun, Y.-N., & Yang, T.-H. (2020). DeepNerve: A New Convolutional Neural Network for the Localization and Segmentation of the Median Nerve in Ultrasound Image Sequences. Ultrasound in Medicine & Biology, 46(9), 2439–2452. https://doi.org/10.1016/j.ultrasmedbio.2020.03.017

Huang, A., Jiang, L., Zhang, J., & Wang, Q. (2022). Attention-VGG16-UNet: A novel deep learning approach for automatic segmentation of the median nerve in ultrasound images. Quantitative Imaging in Medicine and Surgery, 12(6), 3138–3150. https://doi.org/10.21037/qims-21-1074

Huang, C., Zhou, Y., Tan, W., Qiu, Z., Zhou, H., Song, Y., Zhao, Y., & Gao, S. (2019). Applying deep learning in recognizing the femoral nerve block region on ultrasound images. Annals of Translational Medicine, 7. https://doi.org/10.21037/atm.2019.08.61

Jagtap, J. M., Kuroiwa, T., Starlinger, J., Farid, M. H., Lui, H., Akkus, Z., Erickson, B. J., & Amadio, P. (2023). AI for Automated Segmentation and Characterization of Median Nerve Volume. Journal of Medical and Biological Engineering, 43(4), 405–416.

Liu, C., Liu, F., Wang, L., Ma, L., & Lu, Z.-M. (2018). Segmentation of nerve on ultrasound images using deep adversarial network. Int J Innov Comput Inform Control, 14, 53–64.

Ma, J., Kong, D., Wu, F., Bao, L., Yuan, J., & Liu, Y. (2024). Densely connected convolutional networks for ultrasound image based lesion segmentation. Computers in Biology and Medicine, 168, 107725.

Marzola, F., van Alfen, N., Doorduin, J., & Meiburger, K. M. (2021). Deep learning segmentation of transverse musculoskeletal ultrasound images for neuromuscular disease assessment. Computers in Biology and Medicine, 135, 104623. https://doi.org/10.1016/j.compbiomed.2021.104623

Pu, B., Lu, Y., Chen, J., Li, S., Zhu, N., Wei, W., & Li, K. (2022). MobileUNet-FPN: A semantic segmentation model for fetal ultrasound four-chamber segmentation in edge computing environments. IEEE Journal of Biomedical and Health Informatics, 26(11), 5540–5550.

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. In N. Navab, J. Hornegger, W. M. Wells, & A. F. Frangi (Eds.), Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015 (pp. 234–241). Springer International Publishing.

Singarayan, J., Sreelakshmi, C., Ram, K., Rangasami, R., & Sivaprakasam, M. (2020). Artificial intelligence in detection and segmentation of internal auditory canal and its nerves using deep learning techniques. International Journal of Computer Assisted Radiology and Surgery, 15. https://doi.org/10.1007/s11548-020-02237-5

Smistad, E., Iversen, D. H., Leidig, L., Lervik Bakeng, J. B., Johansen, K. F., & Lindseth, F. (2017). Automatic Segmentation and Probe Guidance for Real-Time Assistance of Ultrasound-Guided Femoral Nerve Blocks. Ultrasound in Medicine & Biology, 43(1), 218–226. https://doi.org/10.1016/j.ultrasmedbio.2016.08.036

Smistad, E., Johansen, K., Iversen, D., & Reinertsen, I. (2018). Highlighting nerves and blood vessels for ultrasound-guided axillary nerve block procedures using neural networks. Journal of Medical Imaging, 5, 1. https://doi.org/10.1117/1.JMI.5.4.044004

Wang, J.-C., Shu, Y.-C., Lin, C.-Y., Wu, W.-T., Chen, L.-R., Lo, Y.-C., Chiu, H.-C., Özçakar, L., & Chang, K.-V. (2023). Application of deep learning algorithms in automatic sonographic localization and segmentation of the median nerve: A systematic review and meta-analysis. Artificial Intelligence in Medicine, 137, 102496.

Wu, H., Liu, J., Wang, W., Wen, Z., & Qin, J. (2021). Region-aware Global Context Modeling for Automatic Nerve Segmentation from Ultrasound Images. Proceedings of the AAAI Conference on Artificial Intelligence, 35(4), 2907–2915. https://doi.org/10.1609/aaai.v35i4.16397

X. Ding, X. Zhang, N. Ma, J. Han, G. Ding, & J. Sun. (2021). RepVGG: Making VGG-style ConvNets Great Again. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 13728–13737. https://doi.org/10.1109/CVPR46437.2021.01352
Y. Weng, T. Zhou, Y. Li, & X. Qiu. (2019). NAS-Unet: Neural Architecture Search for Medical Image Segmentation. IEEE Access, 7, 44247–44257. https://doi.org/10.1109/ACCESS.2019.2908991
Zhao, H., & Sun, N. (2017). Improved U-Net Model for Nerve Segmentation. In Y. Zhao, X. Kong, & D. Taubman (Eds.), Image and Graphics (pp. 496–504). Springer International Publishing.


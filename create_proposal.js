import fs from "node:fs";
import path from "node:path";
import {
  AlignmentType, Document, Footer, Header, HeadingLevel, ImageRun,
  ImportedXmlComponent, Packer, PageNumber, Paragraph, ShadingType,
  Table, TableCell, TableRow, TextRun, WidthType, convertInchesToTwip,
} from "docx";

const outputPath = process.argv[2];
if (!outputPath) throw new Error("Usage: node create_proposal.js /absolute/path/output.docx");

const outputDir = path.dirname(outputPath);
const assetDir = path.join(outputDir, "assets");
fs.mkdirSync(assetDir, { recursive: true });

const T = String.raw;

const palette = {
  dark: "263238",
  primary: "37474F",
  light: "78909C",
  border: "D8E0E3",
  fill: "EEF3F6",
};

const font = { name: "Times New Roman", eastAsia: "SimSun" };

const run = (text, options = {}) => new TextRun({ text, font, size: 24, ...options });
const para = (children, options = {}) => new Paragraph({
  spacing: { after: 160, line: 300 },
  ...options,
  children: Array.isArray(children) ? children : [children],
});

const bodyPara = (text, options = {}) => para(run(text), {
  indent: { firstLine: convertInchesToTwip(0.33) },
  ...options,
});

const h1 = (text) => para(run(text, { bold: true, size: 30, color: palette.dark }), {
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 360, after: 200 },
});

const h2 = (text) => para(run(text, { bold: true, size: 26, color: palette.dark }), {
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 280, after: 160 },
});

const h3 = (text) => para(run(text, { bold: true, size: 24, color: palette.primary }), {
  heading: HeadingLevel.HEADING_3,
  spacing: { before: 200, after: 120 },
});

const cell = (text, options = {}) => new TableCell({
  children: [para(run(text))],
  margins: { top: 100, bottom: 100, left: 100, right: 100 },
  ...options,
});

const headerCell = (text, width) => cell(text, {
  shading: { type: ShadingType.CLEAR, fill: palette.fill },
  width: { size: width, type: WidthType.DXA },
});

const xmlEscape = (value) => String(value)
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;");

const toc = (entries) => {
  const cached = entries.map(({ title, level, page }) => {
    const indent = Math.max(0, level - 1) * 360;
    return `<w:p><w:pPr><w:pStyle w:val="TOC${level}"/>
      <w:tabs><w:tab w:val="right" w:leader="dot" w:pos="9000"/></w:tabs>
      <w:ind w:left="${indent}"/></w:pPr>
      <w:r><w:t>${xmlEscape(title)}</w:t></w:r><w:r><w:tab/></w:r><w:r><w:t>${page}</w:t></w:r></w:p>`;
  }).join("");

  return ImportedXmlComponent.fromXmlString(`<w:sdt xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
    <w:sdtPr><w:alias w:val="Table of Contents"/></w:sdtPr>
    <w:sdtContent>
      <w:p><w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/>
        <w:instrText xml:space="preserve"> TOC \\o &quot;1-3&quot; \\h \\z \\u </w:instrText>
        <w:fldChar w:fldCharType="separate"/></w:r></w:p>
      ${cached}
      <w:p><w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>
    </w:sdtContent>
  </w:sdt>`).root[0];
};

const img = (imgPath, caption, width = 500) => {
  const fullPath = path.join(outputDir, imgPath);
  if (!fs.existsSync(fullPath)) {
    return para(run(T`[Image not found: ${imgPath}]`, { italics: true, color: palette.light }));
  }
  const data = fs.readFileSync(fullPath);
  const ext = path.extname(fullPath).toLowerCase();
  const type = ext === ".png" ? "png" : "jpg";
  return [
    para(new ImageRun({ type, data, transformation: { width, height: Math.round(width * 0.6) } })),
    para(run(caption, { italics: true, size: 20, color: palette.light }), { alignment: AlignmentType.CENTER, spacing: { after: 240 } }),
  ];
};

const makeTable = (headers, rows, colWidths) => {
  const widths = colWidths || headers.map(() => 2400);
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    columnWidths: widths,
    rows: [
      new TableRow({ children: headers.map((h, i) => headerCell(h, widths[i])) }),
      ...rows.map(row => new TableRow({
        children: row.map((cellText, i) => cell(cellText, { width: { size: widths[i], type: WidthType.DXA } })),
      })),
    ],
  });
};

// ===== CONTENT =====

const tocEntries = [
  { title: "1. Project Title", level: 1, page: 1 },
  { title: "2. Background and Motivation", level: 1, page: 1 },
  { title: "3. Problem Statement", level: 1, page: 2 },
  { title: "4. Selected Dataset and Link", level: 1, page: 2 },
  { title: "5. Dataset Description", level: 1, page: 2 },
  { title: "6. Research Questions", level: 1, page: 3 },
  { title: "7. Proposed Methodology", level: 1, page: 3 },
  { title: "8. Expected Results", level: 1, page: 5 },
  { title: "9. Actual Results", level: 1, page: 5 },
  { title: "10. Generated Figures and Tables", level: 1, page: 7 },
  { title: "11. Conclusion", level: 1, page: 7 },
  { title: "12. References", level: 1, page: 8 },
];

const children = [];

// Title page
children.push(para(run(T`Facial Expression Recognition using Deep CNN and Transfer Learning on FER2013`, {
  bold: true, size: 36, color: palette.dark
}), { alignment: AlignmentType.CENTER, spacing: { before: 600, after: 200 } }));

children.push(para(run(T`Machine Learning — Phase 2: Proposal and Code Implementation`, {
  size: 26, color: palette.primary
}), { alignment: AlignmentType.CENTER, spacing: { after: 400 } }));

children.push(para(run(T`Table of Contents`, { bold: true, size: 28, color: palette.dark }), {
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 400, after: 200 },
}));

children.push(toc(tocEntries));

// Section 1
children.push(h1("1. Project Title"));
children.push(bodyPara(T`Facial Expression Recognition using Deep CNN and Transfer Learning on FER2013`));

// Section 2
children.push(h1("2. Background and Motivation"));
children.push(bodyPara(T`People express emotions through their faces naturally and constantly. Being able to read these expressions automatically has become important in many real-world areas: healthcare apps that track patient mood, cars that detect if a driver is drowsy or stressed, and online learning platforms that gauge student engagement. The problem is that traditional computer vision methods need hand-crafted features, which do not hold up well when lighting changes, faces turn sideways, or different people show the same emotion in slightly different ways.`));
children.push(bodyPara(T`Convolutional Neural Networks (CNNs) changed this. Instead of manually defining what an "angry eyebrow" or a "happy smile" looks like, CNNs learn these patterns directly from raw image pixels. LeCun et al. (1998) showed early on that CNNs could recognise handwritten digits, and later work by Krizhevsky et al. (2012) proved they could dominate large-scale image classification. For facial expression recognition specifically, CNNs have become the go-to approach because they pick up on subtle texture and shape differences across the face.`));
children.push(bodyPara(T`This project builds a complete facial expression recognition pipeline. I trained a custom CNN from scratch and compared it against two well-known transfer learning models — VGG16 and ResNet50 — to see which approach works best on the FER2013 dataset. The idea was not just to get the highest accuracy, but to understand why one model might outperform another on small, grayscale face images.`));

// Section 3
children.push(h1("3. Problem Statement"));
children.push(bodyPara(T`The task is to classify 48×48 grayscale face images into seven emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. The project covers the full pipeline: loading data, preprocessing, building and training models, evaluating them with standard metrics, and visualising what the models actually learn.`));
children.push(bodyPara(T`Several challenges make this harder than it sounds. First, the dataset is unbalanced — there are over seven thousand Happy faces but only four hundred Disgust faces. Second, some emotions look very similar. Fear and Surprise both involve wide eyes and an open mouth, so telling them apart is difficult even for humans. Third, the images are tiny at 48×48 pixels, so fine details like wrinkles or lip curvature are hard to capture. Finally, transfer learning models were originally trained on ImageNet, which contains large, colourful photos of objects and animals. Adapting them to small grayscale face images is not guaranteed to work well.`));

// Section 4
children.push(h1("4. Selected Dataset and Link"));
children.push(bodyPara(T`Dataset: FER2013 (Facial Expression Recognition 2013)`));
children.push(bodyPara(T`Source: https://www.kaggle.com/datasets/msambare/fer2013`));
children.push(bodyPara(T`The dataset was created by Goodfellow et al. (2013) for the ICML 2013 representation learning challenge. Faces were collected through the Google image search API and automatically registered using a face detection algorithm. It has since become one of the most widely used benchmarks for facial expression recognition research.`));

// Section 5
children.push(h1("5. Dataset Description"));

children.push(h2("5.1 Dataset Properties"));
children.push(makeTable(
  ["Property", "Value"],
  [
    ["Total images", "35,887"],
    ["Training images", "28,709"],
    ["Test images", "7,178"],
    ["Image size", "48 × 48 pixels"],
    ["Image type", "Grayscale"],
    ["Number of classes", "7"],
    ["Format used", "CSV file (fer2013.csv) with pixel strings and Usage labels"],
  ],
  [3600, 3600]
));

children.push(h2("5.2 Class Labels"));
children.push(makeTable(
  ["Label", "Emotion"],
  [
    ["0", "Angry"],
    ["1", "Disgust"],
    ["2", "Fear"],
    ["3", "Happy"],
    ["4", "Sad"],
    ["5", "Surprise"],
    ["6", "Neutral"],
  ],
  [2400, 4800]
));

children.push(h2("5.3 Class Distribution (Training Set)"));
children.push(makeTable(
  ["Emotion", "Count"],
  [
    ["Angry", "3,995"],
    ["Disgust", "436"],
    ["Fear", "4,097"],
    ["Happy", "7,215"],
    ["Sad", "4,830"],
    ["Surprise", "3,171"],
    ["Neutral", "4,965"],
  ],
  [3600, 3600]
));

children.push(bodyPara(T`The imbalance is obvious: Disgust has less than a tenth of the samples that Happy has. This was handled by computing balanced class weights and applying data augmentation during training, which helped the model pay more attention to minority classes.`));

children.push(...img("outputs/01_sample_images.png", "Figure 1: Sample images from each of the seven emotion classes in FER2013", 520));
children.push(...img("outputs/02_class_distribution.png", "Figure 2: Class distribution across training, validation, and test sets", 520));

// Section 6
children.push(h1("6. Research Questions"));
const rqs = [
  T`Can a custom CNN built from scratch achieve competitive accuracy on FER2013 without relying on pre-trained weights?`,
  T`How much does data augmentation improve generalisation and help with the class imbalance problem?`,
  T`Which parts of the face does the model actually look at when making predictions, and does this match what we would expect?`,
  T`Do transfer learning models (VGG16 and ResNet50) outperform the custom CNN when fine-tuned on FER2013?`,
  T`Which emotion pairs get confused most often, and what does this tell us about the model's weaknesses?`,
];
rqs.forEach((rq, i) => {
  children.push(para(run(T`${i + 1}. ${rq}`), { spacing: { after: 120 } }));
});

// Section 7
children.push(h1("7. Proposed Methodology"));

children.push(h2("7.1 Data Loading and Split"));
children.push(bodyPara(T`The dataset was loaded from the fer2013.csv file, which contains three columns: emotion (integer label), pixels (space-separated string of 2,304 pixel values), and Usage (Training, PublicTest, or PrivateTest). The original splits were kept as-is: Training for model fitting, PublicTest for validation during training, and PrivateTest for final evaluation. This gives 28,709 training images, 3,589 validation images, and 3,589 test images. The test set was later combined with the validation set for reporting, giving 7,178 test images in total.`));

children.push(h2("7.2 Preprocessing"));
children.push(bodyPara(T`Each pixel string was parsed into a 48×48 numpy array and reshaped to (48, 48, 1) for the grayscale channel. Pixel values were normalised to the [0, 1] range by dividing by 255. Labels were converted to one-hot encoded vectors with seven classes. For the transfer learning models, grayscale images were simply replicated across three channels since VGG16 and ResNet50 expect RGB input.`));

children.push(h2("7.3 Data Augmentation"));
children.push(bodyPara(T`Augmentation was applied only to the training data through Keras ImageDataGenerator:`));
const augItems = [
  T`Random horizontal flip`,
  T`Random rotation up to 15 degrees`,
  T`Random width and height shifts up to 10%`,
  T`Random zoom up to 10%`,
  T`Nearest-neighbour fill mode for empty pixels`,
];
augItems.forEach(item => children.push(para(run(T`• ${item}`), { spacing: { after: 80 } })));
children.push(bodyPara(T`Validation and test data were left untouched. The augmentation helped create more training variety without collecting new images, which is especially useful for the underrepresented Disgust class.`));

children.push(h2("7.4 Custom CNN Design"));
children.push(bodyPara(T`The custom CNN was designed specifically for 48×48 grayscale face images. It uses three convolutional blocks, each with two convolution layers, batch normalisation, max pooling, and dropout. The idea was to start with a small number of filters (32) to capture basic edges and textures, then increase to 64 and 128 to learn more complex facial features. Batch normalisation stabilised training, while dropout prevented overfitting on this relatively small dataset.`));

children.push(makeTable(
  ["Layer", "Configuration"],
  [
    ["Input", "48 × 48 × 1"],
    ["Conv Block 1", "Conv2D(32) × 2, BatchNorm, MaxPool(2×2), Dropout(0.25)"],
    ["Conv Block 2", "Conv2D(64) × 2, BatchNorm, MaxPool(2×2), Dropout(0.25)"],
    ["Conv Block 3", "Conv2D(128) × 2, BatchNorm, MaxPool(2×2), Dropout(0.25)"],
    ["Classifier", "Flatten → Dense(512) → BatchNorm → Dropout(0.5) → Dense(7, Softmax)"],
  ],
  [2400, 6000]
));

children.push(bodyPara(T`Training setup: Adam optimiser with a learning rate of 0.001, categorical crossentropy loss with balanced class weights, batch size 64, and up to 50 epochs. Early stopping with a patience of 10 epochs was used to prevent overfitting, along with learning rate reduction when validation loss plateaued.`));

children.push(h2("7.5 Transfer Learning Models"));
children.push(bodyPara(T`VGG16: The model was loaded with ImageNet pre-trained weights, excluding the top classification layers. All convolutional base layers were frozen initially, and a custom head was attached: GlobalAveragePooling2D → Dense(512, ReLU) → BatchNorm → Dropout(0.5) → Softmax(7). The model was trained for 15 epochs with the base frozen, then the top four convolutional blocks were unfrozen and fine-tuned for another 20 epochs at a lower learning rate (0.00001).`));
children.push(bodyPara(T`ResNet50: The same setup was used, but the top 10 layers were unfrozen for fine-tuning instead of four. ResNet50's skip connections help with gradient flow in deeper networks, so I expected it might adapt better than VGG16.`));
children.push(bodyPara(T`Both transfer models received the same RGB-converted data and the same class weights as the custom CNN.`));

children.push(h2("7.6 Evaluation Metrics"));
children.push(bodyPara(T`Each model was evaluated on the test set using:`));
const metrics = [
  T`Overall accuracy`,
  T`Macro-averaged precision, recall, and F1-score`,
  T`Per-class precision, recall, and F1-score`,
  T`Normalised confusion matrix`,
  T`ROC curves with AUC (one-vs-rest for each emotion)`,
  T`Training and validation loss/accuracy curves`,
  T`Grad-CAM heatmaps for interpretability`,
  T`Visual error analysis on misclassified samples`,
];
metrics.forEach(m => children.push(para(run(T`• ${m}`), { spacing: { after: 80 } })));

children.push(h2("7.7 Implementation Environment"));
children.push(makeTable(
  ["Component", "Details"],
  [
    ["Framework", "TensorFlow 2.19 / Keras"],
    ["Platform", "Kaggle Notebook with GPU T4 × 2 accelerator"],
    ["Libraries", "NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, OpenCV"],
    ["Code", "Single Python script (facial_expression_recognition.py, 749 lines)"],
  ],
  [3000, 5000]
));

// Section 8
children.push(h1("8. Expected Results"));
children.push(bodyPara(T`Before running the experiments, I expected the custom CNN to reach somewhere between 60% and 70% test accuracy, which is typical for FER2013 in the literature. I also expected the transfer learning models to do at least as well, possibly better, because they had already learned rich visual features from ImageNet. I anticipated that Happy would be the easiest class to recognise due to its large sample size and distinctive smile feature, while Fear and Disgust would be the hardest. I also expected Grad-CAM to show the model focusing on the eyes and mouth, since these are the most expressive parts of the face.`));

// Section 9
children.push(h1("9. Actual Results"));

children.push(h2("9.1 Model Comparison"));
children.push(bodyPara(T`After training all three models and evaluating them on the 7,178 test images, the results were surprising:`));

children.push(makeTable(
  ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "Test Loss"],
  [
    ["Custom CNN", "64.47%", "61.10%", "64.27%", "61.91%", "0.970"],
    ["VGG16", "49.51%", "46.05%", "50.82%", "46.61%", "1.354"],
    ["ResNet50", "32.81%", "28.26%", "31.17%", "26.80%", "1.709"],
  ],
  [2000, 1600, 1600, 1600, 1600, 1600]
));

children.push(bodyPara(T`The custom CNN was the clear winner. It achieved 64.47% accuracy with the lowest test loss (0.970), while VGG16 only managed 49.51% and ResNet50 fell to 32.81%. This was not what I expected. I had assumed that pre-trained models would transfer their learned features effectively, but the gap between ImageNet (large RGB natural images) and FER2013 (tiny grayscale face crops) was apparently too large for the transfer learning setup to bridge with the fine-tuning approach used here.`));

children.push(...img("outputs/03_cnn_training_history.png", "Figure 3: Custom CNN training and validation accuracy/loss curves", 520));

children.push(h2("9.2 Per-Class Results (Custom CNN)"));
children.push(makeTable(
  ["Emotion", "Precision", "Recall", "F1-Score", "Support"],
  [
    ["Angry", "0.54", "0.61", "0.57", "958"],
    ["Disgust", "0.50", "0.71", "0.59", "111"],
    ["Fear", "0.54", "0.38", "0.44", "1,024"],
    ["Happy", "0.89", "0.83", "0.86", "1,774"],
    ["Sad", "0.55", "0.44", "0.49", "1,247"],
    ["Surprise", "0.72", "0.82", "0.77", "831"],
    ["Neutral", "0.54", "0.70", "0.61", "1,233"],
  ],
  [2000, 1600, 1600, 1600, 1600]
));

children.push(bodyPara(T`Happy was by far the easiest class, with an F1-score of 0.86. This makes sense: it has the most training examples, and a smile is visually distinct from the other emotions. Surprise also performed well (F1 = 0.77), probably because raised eyebrows and an open mouth create a strong, recognisable pattern.`));
children.push(bodyPara(T`Fear was the weakest class, with a recall of only 0.38. This means the model missed nearly two-thirds of all fear images, likely confusing them with Sad or Surprise. Disgust had decent recall (0.71) despite having the fewest samples, which suggests the class weights and augmentation did help. However, its precision was only 0.50, meaning the model sometimes labelled other emotions as Disgust when it was unsure.`));

children.push(...img("outputs/04_cnn_confusion_matrix.png", "Figure 4: Custom CNN normalised confusion matrix", 480));
children.push(...img("outputs/05_cnn_roc_curves.png", "Figure 5: Custom CNN ROC curves with per-class AUC", 480));

children.push(h2("9.3 Research Question Answers"));

children.push(h3("RQ1 — Can a custom CNN achieve competitive accuracy without pre-trained weights?"));
children.push(bodyPara(T`Yes. The custom CNN reached 64.47% accuracy, which falls within the typical range reported for FER2013 in the literature (Mollahosseini et al., 2016; Barsoum et al., 2016). This shows that a carefully designed architecture, trained with proper regularisation and class balancing, can perform well even without transfer learning.`));

children.push(h3("RQ2 — How does data augmentation affect training?"));
children.push(bodyPara(T`Augmentation and class weighting together kept training stable and improved performance on minority classes. Without them, the model would likely have ignored Disgust entirely and overfit to Happy. The validation curves stayed close to the training curves, indicating that augmentation helped generalisation rather than just memorisation.`));

children.push(h3("RQ3 — Which facial regions does the model focus on?"));
children.push(bodyPara(T`Grad-CAM heatmaps showed the custom CNN concentrating on the eye region and the mouth area when making predictions. This aligns with psychological research on facial expressions: the eyes carry a lot of emotional information (especially for Fear and Surprise), while the mouth distinguishes Happy from Neutral or Sad.`));
children.push(...img("outputs/gradcam_cnn/gradcam_CustomCNN_0.png", "Figure 6: Grad-CAM visualisation showing model attention on eyes and mouth", 420));
children.push(...img("outputs/gradcam_cnn/gradcam_CustomCNN_2.png", "Figure 7: Grad-CAM visualisation on a Surprise example", 420));

children.push(h3("RQ4 — Do transfer learning models outperform the custom CNN?"));
children.push(bodyPara(T`No. VGG16 and ResNet50 both underperformed significantly. VGG16 at 49.51% was 15 percentage points behind the custom CNN, and ResNet50 at 32.81% was barely better than random guessing (which would be ~14%). Several factors probably contributed: the 48×48 input is much smaller than what these models were designed for, grayscale images lack the colour information ImageNet models expect, and the two-phase fine-tuning with limited epochs may not have been enough to adapt the deep pre-trained features to face-specific patterns.`));

children.push(...img("outputs/06_vgg_training_history.png", "Figure 8: VGG16 training history curves", 480));
children.push(...img("outputs/06_vgg_confusion_matrix.png", "Figure 9: VGG16 normalised confusion matrix", 480));
children.push(...img("outputs/07_vgg_roc_curves.png", "Figure 10: VGG16 ROC curves with per-class AUC", 480));
children.push(...img("outputs/08_resnet_training_history.png", "Figure 11: ResNet50 training history curves", 480));
children.push(...img("outputs/08_resnet_confusion_matrix.png", "Figure 12: ResNet50 normalised confusion matrix", 480));
children.push(...img("outputs/09_resnet_roc_curves.png", "Figure 13: ResNet50 ROC curves with per-class AUC", 480));

children.push(h3("RQ5 — Which emotion pairs are most confused?"));
children.push(bodyPara(T`The confusion matrix revealed that Fear was frequently misclassified as Sad and Surprise. Sad was also confused with Neutral. Happy had the cleanest diagonal, meaning it was rarely mistaken for something else. These patterns match human intuition: Fear, Sad, and Surprise all involve some form of mouth opening or eye widening, while Happy stands apart with its upward lip curve.`));

children.push(h2("9.4 Key Findings"));
const findings = [
  T`A task-specific custom CNN can outperform generic pre-trained models when the target domain differs significantly from the source domain.`,
  T`Happy is the most reliably recognised emotion, while Fear is the most difficult.`,
  T`Class weighting and augmentation are essential for handling imbalance in FER2013.`,
  T`Transfer learning from ImageNet to small grayscale face images requires more aggressive fine-tuning or domain-specific pre-training to be effective.`,
  T`All required visualisations were generated successfully: training curves, confusion matrices, ROC curves, Grad-CAM overlays, error analysis, and a comparison chart.`,
];
findings.forEach(f => children.push(para(run(T`• ${f}`), { spacing: { after: 100 } })));

children.push(...img("outputs/10_model_comparison.png", "Figure 14: Model comparison bar chart across all three architectures", 520));
children.push(...img("outputs/11_error_analysis.png", "Figure 15: Error analysis showing misclassified test samples", 520));

// Section 10
children.push(h1("10. Generated Figures and Tables"));
children.push(makeTable(
  ["Output", "Description", "File"],
  [
    ["Sample images", "One example from each of the 7 classes", "01_sample_images.png"],
    ["Class distribution", "Bar chart showing train/val/test splits", "02_class_distribution.png"],
    ["Custom CNN training curves", "Accuracy and loss over epochs", "03_cnn_training_history.png"],
    ["Custom CNN confusion matrix", "Normalised class-wise predictions", "04_cnn_confusion_matrix.png"],
    ["Custom CNN ROC curves", "One-vs-rest AUC for each emotion", "05_cnn_roc_curves.png"],
    ["VGG16 training curves", "Accuracy and loss over epochs", "06_vgg_training_history.png"],
    ["VGG16 confusion matrix", "Normalised class-wise predictions", "06_vgg_confusion_matrix.png"],
    ["VGG16 ROC curves", "One-vs-rest AUC for each emotion", "07_vgg_roc_curves.png"],
    ["ResNet50 training curves", "Accuracy and loss over epochs", "08_resnet_training_history.png"],
    ["ResNet50 confusion matrix", "Normalised class-wise predictions", "08_resnet_confusion_matrix.png"],
    ["ResNet50 ROC curves", "One-vs-rest AUC for each emotion", "09_resnet_roc_curves.png"],
    ["Model comparison chart", "Bar chart comparing all three models", "10_model_comparison.png"],
    ["Error analysis", "Misclassified examples from the custom CNN", "11_error_analysis.png"],
    ["Model comparison table", "CSV with numerical results", "model_comparison.csv"],
    ["Grad-CAM visualisations", "Heatmaps overlaid on test images", "gradcam_cnn/"],
    ["Classification reports", "Text files with per-class metrics", "*_classification_report.txt"],
  ],
  [2400, 4200, 2400]
));

// Section 11
children.push(h1("11. Conclusion"));
children.push(bodyPara(T`This project successfully built and evaluated a facial expression recognition system on the FER2013 dataset. The custom CNN achieved 64.47% test accuracy and outperformed both VGG16 and ResNet50, which was unexpected but informative. It suggests that for small, grayscale, domain-specific images like facial expressions, a carefully designed custom architecture can be more effective than simply applying transfer learning from a very different visual domain.`));
children.push(bodyPara(T`The results also highlight the ongoing challenges in facial expression recognition: class imbalance, subtle inter-class similarities, and the difficulty of transferring knowledge across image domains. Happy and Surprise were recognised reliably, but Fear remained difficult, often being confused with Sad or Surprise.`));
children.push(bodyPara(T`For future work, I would consider exploring lightweight architectures like EfficientNet or MobileNet, which are designed to be more parameter-efficient. Another direction would be to use facial landmark detection as an additional input channel, giving the model explicit information about eye and mouth positions. Finally, training the transfer learning models for more epochs with a more gradual unfreezing strategy might help them adapt better to the FER2013 domain.`));

// Section 12
children.push(h1("12. References"));
const refs = [
  T`Barsoum, E., Zhang, C., Ferrer, C. C., & Zhang, Z. (2016). Training deep networks for facial expression recognition with crowd-sourced label distribution. Proceedings of the 18th ACM International Conference on Multimodal Interaction, 279–283. https://doi.org/10.1145/2993148.2993165`,
  T`Goodfellow, I. J., Erhan, D., Carrier, P. L., Courville, A., Mirza, M., Hamner, B., Cukierski, W., Tang, Y., Thaler, D., Lee, D. H., Zhou, Y., Ramaiah, C., Feng, F., Li, R., Wang, X., Athanasakis, D., Shawe-Taylor, J., Milakov, M., Park, J., Ionescu, R. T., Popescu, M., Grozea, C., Bergstra, J., Xie, J., Romaszko, L., Xu, B., Chuang, Z., & Bengio, Y. (2013). Challenges in representation learning: A report on three machine learning contests. Neural Information Processing, 117–124. https://doi.org/10.1007/978-3-642-42051-1_16`,
  T`He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770–778. https://doi.org/10.1109/CVPR.2016.90`,
  T`Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. Advances in Neural Information Processing Systems (NeurIPS), 25, 1097–1105. https://doi.org/10.1145/3065386`,
  T`LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278–2324. https://doi.org/10.1109/5.726791`,
  T`Mollahosseini, A., Hasani, B., & Mahoor, M. H. (2016). AffectNet: A database for facial expression, valence, and arousal computing in the wild. IEEE Transactions on Affective Computing, 10(1), 18–31. https://doi.org/10.1109/TAFFC.2017.2740923`,
  T`Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. Proceedings of the IEEE International Conference on Computer Vision (ICCV), 618–626. https://doi.org/10.1109/ICCV.2017.74`,
  T`Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. Journal of Big Data, 6(1), 1–48. https://doi.org/10.1186/s40537-019-0197-0`,
  T`Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556. https://arxiv.org/abs/1409.1556`,
  T`Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(1), 1929–1958. https://jmlr.org/papers/v15/srivastava14a.html`,
  T`Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How transferable are features in deep neural networks? Advances in Neural Information Processing Systems (NeurIPS), 27. https://proceedings.neurips.cc/paper/2014/hash/375c71349b295fbe208edf62c5c5f0f6-Abstract.html`,
];
refs.forEach((ref, i) => {
  children.push(para(run(T`[${i + 1}] ${ref}`), { spacing: { after: 120, line: 280 } }));
});

// Build document
const doc = new Document({
  features: { updateFields: true },
  sections: [{
    properties: {
      page: {
        margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 },
      },
    },
    headers: {
      default: new Header({
        children: [para(run(T`Facial Expression Recognition — Phase 2 Proposal`, { bold: true, color: palette.primary, size: 20 }), {
          alignment: AlignmentType.CENTER,
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [para(new TextRun({ children: [PageNumber.CURRENT] }), {
          alignment: AlignmentType.CENTER,
        })],
      }),
    },
    children,
  }],
});

const buffer = await Packer.toBuffer(doc);
fs.writeFileSync(outputPath, buffer);
console.log("Document written to:", outputPath);

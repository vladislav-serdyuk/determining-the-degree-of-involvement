# Научные источники и обоснование

Параметры и веса `EngagementCalculator` подобраны не произвольно, а с опорой на академические публикации по эмоциям, усталости и оценке внимания. Список ниже фиксирует **не все, но ключевые** из них.

---

## Публикации

### Эмоции

#### **Buono, P., De Carolis, B., D'Errico, F., Macchiarulo, N., & Palestra, G. (2022, опубл. онлайн; MTA 2023).** Assessing student engagement from facial behavior in on-line learning. *Multimedia Tools and Applications*, 82(9), 12859–12877. DOI: [10.1007/s11042-022-14048-8](https://doi.org/10.1007/s11042-022-14048-8). PMC: PMC9589763.

Эмпирическое исследование: LSTM-модель обучается на данных EAR, позы головы и лицевых экспрессий в условиях онлайн-обучения. Корреляция предсказания с эмоциональным компонентом самоотчётной вовлечённости – ρ ≈ 0.38 (наибольшая среди трёх модальностей). Позитивные экспрессии (`Happiness`, `Surprise`) ассоциированы с высокой вовлечённостью, `Anger` и `Disgust` – с отторжением материала.

Применение в проекте:
- `EMOTION_WEIGHTS` в `engagement_calculator.py` ранжированы по выводам работы.
- `Surprise = 0.8` (не 1.0): удивление продуктивно при новизне, но при повторении переходит в негативную валентность.

---

### Состояние глаз (EAR)

#### **Soukupová, T., & Čech, J. (2016).** Real-Time Eye Blink Detection using Facial Landmarks. *21st Computer Vision Winter Workshop*, Rimske Toplice, Slovenia.

Первоисточник формулы EAR на 6 точках глазного контура:

```
EAR = (||P2−P6|| + ||P3−P5||) / (2 × ||P1−P4||)
```

#### **Dewi, C., Chen, R.-C., Chang, C.-W., Wu, S.-H., Jiang, X., & Yu, H. (2022).** Eye Aspect Ratio for Real-Time Drowsiness Detection to Improve Driver Safety. *Electronics*, 11(19), 3183. DOI: [10.3390/electronics11193183](https://doi.org/10.3390/electronics11193183).

Снижение EAR ниже пороговых значений надёжно сигнализирует об усталости у водителей (точность системы 94.9%). Предложенная пороговая шкала адаптирована в проекте.

---

### Частота моргания

#### **Magliacano, A., Fiorenza, S., Estraneo, A., & Trojano, L. (2020).** Eye blink rate increases as a function of cognitive load during an auditory oddball paradigm. *Neuroscience Letters*, 736, **135293**. DOI: [10.1016/j.neulet.2020.135293](https://doi.org/10.1016/j.neulet.2020.135293).

Статья показывает, что частота спонтанных морганий (EBR) **растёт с когнитивной нагрузкой** и что EBR не коррелирует с амплитудой/латентностью P300. Используется в проекте как обоснование зависимости `blink_modifier` от частоты морганий: отклонение от нормы сигнализирует об изменении когнитивного состояния.

#### **Bentivoglio, A. R., Bressman, S. B., Cassetta, E., Carretta, D., Tonali, P., & Albanese, A. (1997).** Analysis of blink rate patterns in normal subjects. *Movement Disorders*, 12(6), 1028–1034. DOI: [10.1002/mds.870120629](https://doi.org/10.1002/mds.870120629).

Первичный источник нормативного диапазона спонтанной частоты моргания: **15–20 раз/мин** (доверительный интервал охватывает 10–25/мин). Диапазон "10–25/мин" в коде проекта – округлённая клиническая норма из этой работы.

#### **Jongkees, B. J., & Colzato, L. S. (2016).** Spontaneous eye blink rate as predictor of dopamine-related cognitive function – a review. *Neuroscience & Biobehavioral Reviews*, 71, 58–82. DOI: [10.1016/j.neubiorev.2016.08.020](https://doi.org/10.1016/j.neubiorev.2016.08.020).

Обзор связывает высокую частоту моргания (>30/мин) с дофаминергической активацией при стрессе и тревожности. Обосновывает штрафной модификатор `x0.90` при `rate > 30`.

Применение:
- `10–25` морг/мин → `x1.10` (норма бодрствования) – Bentivoglio et al. 1997
- `< 5` морг/мин → `x0.95` (гиперфокус или начало усталости) – Magliacano et al. 2020
- `> 30` морг/мин → `x0.90` (стресс, дофаминергическая активация) – Jongkees & Colzato 2016
- Подробности: [modifiers.md](modifiers.md#blink-rate-modifier-для-eye_score).

---

### Поза головы

#### **Raca, M., Kidzinski, Ł., & Dillenbourg, P. (2015).** Translating Head Motion into Attention – Towards Processing of Student's Body-Language. *Proceedings of the 8th International Conference on Educational Data Mining (EDM 2015)*, Madrid, Spain.

Синхронизация ориентации головы студента с движениями преподавателя – надёжный индикатор внимания в условиях класса, применимый и к условиям применения гаджетов, видеосистем. Обосновывает включение модуля HPE как вспомогательного компонента вовлечённости.

> Roll в классификацию не включён: Raca et al. показывают, что боковой наклон головы слабо коррелирует с внимательностью (учащиеся часто опираются на руку без потери фокуса). Согласуется с выводами Sümer et al. (2021).

#### **Sümer, Ö., Goldberg, P., D'Mello, S., Gerjets, P., Trautwein, U., & Kasneci, E. (2021).** Multimodal Engagement Analysis from Facial Videos in the Classroom. *IEEE Transactions on Affective Computing*, arXiv:2101.04215.

Мультимодальная система (FER + HPE + gaze) в условиях реального класса: поза головы вносит вспомогательный, но значимый вклад в предсказание вовлечённости. Обосновывает меньший вес HPE-компонента относительно эмоций и EAR.

Пороговые значения углов (`|pitch| < 10°/20°/30°`, `|yaw| < 15°/25°/40°`) в [`classify_attention_state`](../../backend/app/services/video_processing/analyze_head_pose.py#L160) – выбор, согласующийся с диапазонами в работах по оценке внимания через позу головы:

#### **Zaletelj, J., & Košir, A. (2017).** Predicting students' attention in the classroom from Kinect facial and body features. *EURASIP Journal on Image and Video Processing*, 2017(1), 80. DOI: [10.1186/s13640-017-0228-8](https://doi.org/10.1186/s13640-017-0228-8).

---

### Perspective-n-Point

#### **Lepetit, V., Moreno-Noguer, F., & Fua, P. (2009).** EPnP: An accurate O(n) solution to the PnP problem. *International Journal of Computer Vision*, 81(2), 155–166. DOI: [10.1007/s11263-008-0152-6](https://doi.org/10.1007/s11263-008-0152-6).

Алгоритм EPnP с линейной сложностью по числу точек для решения задачи Perspective-n-Point. В проекте PnP используется при 6 точках и упрощённой pinhole-модели камеры.

---

### Face Mesh

#### **Kartynnik, Y., Ablavatski, A., Grishchenko, I., & Grundmann, M. (2019).** Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs. *CVPR Workshop on Computer Vision for AR/VR*. arXiv:1907.06724.

Первоисточник базовой Face Mesh с **468 landmarks** в MediaPipe. Топология сетки используется в проекте для вычисления EAR (точки глаз) и HPE (6 опорных ландмарков).

#### **Grishchenko, I., Ablavatski, A., Kartynnik, Y., Raveendran, K., & Grundmann, M. (2020).** Attention Mesh: High-fidelity Face Mesh Prediction in Real-time. *CVPR Workshop on Computer Vision for Augmented and Virtual Reality (CV4ARVR)*. arXiv:2006.10962.

Расширение Face Mesh: attention-головы для глаз, губ и радужки поверх 468-точечной сетки.

---

### Распознавание эмоций

#### **Savchenko, A. V. (2022).** Video-based Frame-level Facial Analysis of Affective Behavior on Mobile Devices using EfficientNets. 2358-2365. 10.1109/CVPRW56347.2022.00263. 

Семейство моделей `enet_b0/b1/b2/b3` (EfficientNet), предобученных на VGGFace2 и дообученных на AffectNet-8. Библиотека EmotiEffLib: https://github.com/sb-ai-lab/EmotiEffLib.

#### **Savchenko, A. (2023).** Facial Expression Recognition with Adaptive Frame Rate based on Multiple Testing Correction. *Proceedings of the 40th International Conference on Machine Learning*, PMLR 202:30119-30129.

Адаптивная частота кадров через процедуру Бенджамини–Хохберга для контроля FDR между соседними кадрами. Обоснование `bypass_threshold` в адаптивном сглаживании `EngagementCalculator`: при стабильном состоянии (низкая дисперсия) достаточно короткого окна, при нестабильном – полного.

---

## Пороги классификации вовлечённости: [0.25; 0.50; 0.75]

Пороги `High ≥ 0.75`, `Medium ≥ 0.50`, `Low ≥ 0.25`, `Very Low < 0.25` – эквидистантное квартильное отображение непрерывной оценки вовлечённости на четырёхуровневую порядковую шкалу. Данный выбор согласуется с рядом академических источников ниже.

### Четырёхуровневая шкала в датасетах

#### **Gupta, A., D'Cunha, A., Awasthi, K., & Balasubramanian, V. (2016).** DAiSEE: Towards User Engagement Recognition in the Wild. arXiv:1609.01885.

Использует четыре уровня вовлечённости *Very Low / Low / High / Very High*, размеченные психологами на краудсорсинге.

#### **Kaur, A., Mustafa, A., Mehta, L., & Dhall, A. (2018).** Prediction and Localization of Student Engagement in the Wild. *IEEE Digital Image Computing: Techniques and Applications (DICTA 2018)*, pp. 1–8. arXiv:1804.00858.

Также независимо вводит четырёхуровневую шкалу *Disengaged / Barely-engaged / Engaged / Highly-engaged* для датасета EngageWild. Использовалась в EmotiW Engagement Prediction Challenge 2018–2020.

#### **Whitehill, J., Serpell, Z., Lin, Y.-C., Foster, A., & Movellan, J. R. (2014).** The Faces of Engagement: Automatic Recognition of Student Engagement from Facial Expressions. *IEEE Transactions on Affective Computing*, 5(1), 86–98. DOI: [10.1109/TAFFC.2014.2316163](https://doi.org/10.1109/TAFFC.2014.2316163).

Устанавливает 4 порядковых уровня вовлечённости (от 1 "not engaged" до 4 "very engaged") со статистически значимым межэкспертным согласием. Подтверждает экспериментально, что именно четыре уровня устойчиво различимы независимыми наблюдателями-людьми (κ > 0.6).


### Отображение [0,1] на четыре класса – EmotiW Challenge

#### **Dhall, A., Kaur, A., Goecke, R., & Gedeon, T. (2018).** EmotiW 2018: Audio-Video, Student Engagement and Group-Level Affect Prediction. *Proceedings of the 20th ACM International Conference on Multimodal Interaction (ICMI 2018)*, pp. 653–656. DOI: [10.1145/3242969.3264993](https://doi.org/10.1145/3242969.3264993).

В Engagement Prediction sub-challenge целевая переменная – непрерывное значение в **[0, 1]**, квантуется на четыре уровня: 0 → *Very Low*, ~0.33 → *Low*, ~0.66 → *High*, 1.0 → *Very High*. Пороги 0.25 / 0.50 / 0.75 в проекте – границы между этими уровнями, округлённые до симметричных квартилей. Это упрощает интерпретацию без потери согласованности с формулировкой EmotiW.

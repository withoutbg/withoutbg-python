# Third-Party Licenses

This project uses the following third-party components.

The withoutBG Open Weights Model is a composite artifact. withoutBG-authored
portions are Apache 2.0; DINOv3 backbone weights embedded in the model are
subject to the Meta DINOv3 License. See the
[withoutBG Open Model License](https://withoutbg.com/open-weights-model/license)
for the combined terms.

## DINOv3 (Meta Platforms)

**License**: Meta DINOv3 License  
**License URL**: https://ai.meta.com/resources/models-and-libraries/dinov3-license/  
**Usage**: DINOv3 ViT backbone weights embedded in the withoutBG Open Weights Model  
**Attribution**: Built with DINOv3

Full license text: [LICENSE-DINOv3](LICENSE-DINOv3)

When you distribute the Open Weights Model or derivatives that contain DINOv3
Materials, you must:

1. Include a copy of the Meta DINOv3 License (`LICENSE-DINOv3`)
2. Prominently display "Built with DINOv3" in related product documentation,
   app About screens, or model cards
3. Comply with Meta's trade-control and prohibited-use restrictions

## Depth Anything V2

**Repository**: https://github.com/DepthAnything/Depth-Anything-V2  
**License**: Apache License 2.0  
**Usage**: Used as part of the background removal pipeline (Small model only)

```
Copyright (c) 2024 Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, Hengshuang Zhao

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## ISNet (Highly Accurate Dichotomous Image Segmentation)

**Repository**: https://github.com/xuebinqin/DIS  
**License**: Apache License 2.0  
**Usage**: Used as part of the background removal pipeline for segmentation

```
Copyright (c) 2022 Xuebin Qin, Hang Dai, Xiaobin Hu, Deng-Ping Fan, Ling Shao, Luc Van Gool

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## segmentation-models-pytorch

**Repository**: https://github.com/qubvel/segmentation_models.pytorch  
**License**: MIT License  
**Usage**: Library used to train the matting and refiner models (Not distributed with this project)

```
Copyright (c) 2019 Pavel Yakubovskiy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Python Dependencies

The following Python packages are used by this project. Please refer to their respective licenses:

- **numpy**: BSD License
- **pillow**: PIL License (MIT-style)
- **onnxruntime**: MIT License
- **requests**: Apache License 2.0
- **click**: BSD License
- **tqdm**: MIT License
- **huggingface-hub**: Apache License 2.0

For complete license text of dependencies, use:
```bash
pip-licenses --format=markdown --output-file=DEPENDENCY_LICENSES.md
```

## Attribution Requirements

Any distribution of this software or the Open Weights Model must:

1. Include a copy of the Apache License 2.0 (`LICENSE` file)
2. Include this `THIRD_PARTY_LICENSES.md` file
3. Include a copy of the Meta DINOv3 License (`LICENSE-DINOv3`) when distributing
   model weights or derivatives that embed DINOv3 Materials
4. Include the `NOTICE` file
5. Retain all copyright notices from the original authors
6. Include attribution to withoutbg.com in any public usage
7. Display "Built with DINOv3" when distributing the Open Weights Model, as
   required by the Meta DINOv3 License

## Model Weights Attribution

The withoutBG Open Weights Model ONNX graph embeds DINOv3-derived weights and
components from the repositories listed above. Users redistributing these weights
must comply with the
[withoutBG Open Model License](https://withoutbg.com/open-weights-model/license),
including Apache 2.0 for withoutBG portions and the Meta DINOv3 License for
DINOv3 portions.

---

For questions about licensing, contact: contact@withoutbg.com
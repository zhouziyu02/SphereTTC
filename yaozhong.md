

## 只需要看两个文件

| 要画的 | 文件 | 从哪里读起 |
|---|---|---|
| **SphereTTC 模块**（我们提出的 test-time calibration） | [`src/spherettc.py`](src/spherettc.py) | 文件头 docstring 是完整的数据流。核心类是 `SphereTTCCalibrator`，在线插件是 `SphereTTC`。 |
| **SphereDyn 模型**（我们提出的 backbone） | [`src/spheredyn.py`](src/spheredyn.py) | 文件头 docstring 是完整的数据流。主类是 `SphereDyn`，入口 `SphereDyn.forward`。 |

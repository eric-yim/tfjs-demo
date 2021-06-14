# Bringing Face Styling to the Browser
![demo](style-face-video.gif)

## How was this built?
The javascript is based on the Tensorflow.js PoseNet demo. PoseNet is also used to center a person's face.
https://github.com/tensorflow/tfjs-models/tree/master/posenet/demo

The main model, is trained via StarGAN. https://github.com/clovaai/stargan-v2

## Start up instructions

Cd into the posenet folder:
```sh
cd posenet
yarn
yarn build && yarn yalc publish
```

Cd into the demo and install dependencies:

```sh
cd demo
yarn
yarn yalc link @tensorflow-models/posenet
```

Start the dev demo server:
```sh
yarn start
```


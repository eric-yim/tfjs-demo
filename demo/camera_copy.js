import * as posenet from '@tensorflow-models/posenet';
//import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';
import * as tfconv from '@tensorflow/tfjs-converter';
import {drawBoundingBox, drawKeypoints, drawSkeleton, isMobile, toggleLoadingUI, tryResNetButtonName, tryResNetButtonText, updateTryResNetButtonDatGuiCss} from './demo_util';
import * as tf from '@tensorflow/tfjs-core';
/**
 * Posenet Config
 */
const defaultQuantBytes = 2;

const defaultMobileNetMultiplier = isMobile() ? 0.50 : 0.75;
const defaultMobileNetStride = 16;
const defaultMobileNetInputResolution = 200;
var repositionFrame = 0;
const repositionInterval = 10;
const guiState = {
  algorithm: 'single-pose',
  input: {
    architecture: 'MobileNetV1',
    outputStride: defaultMobileNetStride,
    inputResolution: defaultMobileNetInputResolution,
    multiplier: defaultMobileNetMultiplier,
    quantBytes: defaultQuantBytes
  },
  singlePoseDetection: {
    minPoseConfidence: 0.1,
    minPartConfidence: 0.5,
  },
  multiPoseDetection: {
    maxPoseDetections: 5,
    minPoseConfidence: 0.15,
    minPartConfidence: 0.1,
    nmsRadius: 30.0,
  },
  output: {
    showVideo: true,
    showSkeleton: true,
    showPoints: true,
    showBoundingBox: false,
  },
  net: null,
};

/**
 * Loads a the camera to be used in the demo
 *
 */
const videoWidth = 600;
const videoHeight = 500;
async function setupCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error(
          'Browser API navigator.mediaDevices.getUserMedia not available');
    }
  
    const video = document.getElementById('video');
    video.width = videoWidth;
    video.height = videoHeight;
  
    const mobile = isMobile();
    const stream = await navigator.mediaDevices.getUserMedia({
      'audio': false,
      'video': {
        facingMode: 'user',
        width: mobile ? undefined : videoWidth,
        height: mobile ? undefined : videoHeight,
      },
    });
    video.srcObject = stream;
    return new Promise((resolve) => {
      video.onloadedmetadata = () => {
        resolve(video);
      };
    });
    
}
  
async function loadVideo() {
    const video = await setupCamera();
    video.play();
  
    return video;
}
/**
 * Feeds an image to posenet to estimate poses - this is where the magic
 * happens. This function loops with a requestAnimationFrame method.
 */
function detectPoseInRealTime(video, net,styleGenerator) {
    const showPosenet = false;
    
    const styleSelect = document.getElementById('style-name');
    var styleImage;
    //var sourceImage = loadImageById('source-input');
    var sourceImage,outputImage;
    var styleDomain;
    var styleSelected;
    
    const canvas = document.getElementById('output');
    //const canvasStyle = document.getElementById('style-image');
    canvas.width = videoWidth;
    canvas.height = videoHeight;
    const ctx = canvas.getContext('2d');

    
    var tempBoxes = [0,0,0,0];
    var hasPassed = false;
    async function poseDetectionFrame() {
        const flipPoseHorizontal = true;
        let poses = [];
        let minPoseConfidence;
        let minPartConfidence;
        let img, boxes, boxInd, cropSize;
        const pose = await net.estimatePoses(video, {
            flipHorizontal: flipPoseHorizontal,
            decodingMethod: 'single-person'
        });
        poses = poses.concat(pose);
        minPoseConfidence = +guiState.singlePoseDetection.minPoseConfidence;
        minPartConfidence = +guiState.singlePoseDetection.minPartConfidence;
        if (repositionFrame>repositionInterval){
            hasPassed=false;
            repositionFrame=0;
        }
        
        
        

        if (showPosenet){
            ctx.clearRect(0, 0, videoWidth, videoHeight);
            
            if (guiState.output.showVideo) {
            ctx.save();
            ctx.scale(-1, 1);
            ctx.translate(-videoWidth, 0);
            
            ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
            ctx.restore();
            }
            
            // For each pose (i.e. person) detected in an image, loop through the poses
            // and draw the resulting skeleton and keypoints if over certain confidence
            // scores
            poses.forEach(({score, keypoints}) => {
                
                if (score >= minPoseConfidence) {
                if (guiState.output.showPoints) {
                    drawKeypoints(keypoints, minPartConfidence, ctx);
                }
                if (guiState.output.showSkeleton) {
                    drawSkeleton(keypoints, minPartConfidence, ctx);
                }
                if (guiState.output.showBoundingBox) {
                    drawBoundingBox(keypoints, ctx);
                }
                

                }
            });
        }else{
            if (!hasPassed){
            
                poses.forEach(({score, keypoints}) => {
                    
                    if (score >= minPoseConfidence) {
                        [hasPassed,tempBoxes] = scaleByEyesNose(keypoints, minPartConfidence,videoWidth,videoHeight);
                    }
                    
                });
            }
            if (hasPassed){
                showSelected();
                tf.engine().startScope()
                styleSelected = document.getElementById('style-'+styleSelect.value)
                styleImage = tf.browser.fromPixels(styleSelected)
                styleImage = formatStyle(styleImage);
                sourceImage = tf.div(tf.expandDims(tf.browser.fromPixels(video),0),255.0);
                boxes = tf.tensor([tempBoxes]);
                boxInd = tf.tensor([0],[1],'int32');
                cropSize = [256,256];
                sourceImage = tf.image.cropAndResize(tf.image.flipLeftRight(sourceImage),boxes,boxInd,cropSize);
                sourceImage = adjustDynamicRange(sourceImage,0.0,1.0,-1.0,1.0);
                styleDomain =  tf.tensor([[styleSelected.getAttribute("gender")]],[1,1],'int32');// 0 female, 1 male?
                //styleCode = styleEncoder.predict([styleImage,styleDomain]);
                // //Quantized Model input order may have changed during TFJS converter stage
                outputImage = formatOutput(styleGenerator.predict([styleImage,styleDomain,sourceImage]));
                // outputImage = formatOutput(generator.predict([sourceImage,styleCode]));
                
                tf.browser.toPixels(outputImage,canvas);
                tf.engine().endScope()

            }
            repositionFrame = repositionFrame+1;
            

        }
        requestAnimationFrame(poseDetectionFrame);
    }
    poseDetectionFrame();
    

    //tf.browser.toPixels(img,canvas);

}
/**
 * Crop face to square
 */

function scaleByEyesNose(keypoints, minConfidence,videoWidth,videoHeight,scale=3.7) {
    //Put mid point of eyes right in the center
    //Use eye-nose-y-value as unit of measure
    //scale*eyedist in each direction
    //Right Eye
    if (keypoints[2].score < minConfidence){
        return [false, [0,0,0,0]];
    }
    //Left Eye
    if (keypoints[1].score < minConfidence){
        return [false, [0,0,0,0]];
    }
    //Nose
    if (keypoints[0].score < minConfidence){
        return [false, [0,0,0,0]];
    }
    
    const midpoint_x = (keypoints[1].position.x +  keypoints[2].position.x)/2.0;
    const midpoint_y = (keypoints[1].position.y +  keypoints[2].position.y)/2.0;
    const unit_dist = keypoints[0].position.y -midpoint_y;
    const left = midpoint_x - (scale * unit_dist);
    const right = midpoint_x + (scale*unit_dist);
    const top = midpoint_y - (scale * unit_dist*0.95);
    const bottom = midpoint_y +(scale*unit_dist*1.05);
    return [true,[top/videoHeight,left/videoWidth,bottom/videoHeight,right/videoWidth]];

}

/**
 * Load Image and Scale 0-255 to -1-1
 * ExpandsDims for Inference
 */
function adjustDynamicRange(image,rangeInMin,rangeInMax,rangeOutMin,rangeOutMax){
    const scale = tf.tensor((rangeOutMax-rangeOutMin)/(rangeInMax-rangeInMin));
    const bias =  tf.sub(rangeOutMin, tf.mul(rangeInMin ,scale));
    var image = tf.add(tf.mul(image,scale),bias);
    image = tf.clipByValue(image,rangeOutMin,rangeOutMax);
    return image
}
function formatOutput(image){
    var image = tf.cast(adjustDynamicRange(image,-1.0,1.0,0.0,255.0),'int32');
    return tf.squeeze(image,0);
}

function formatStyle(img){
    var image = tf.expandDims(tf.image.resizeBilinear(img,[256,256]),0);
    image = adjustDynamicRange(image,0.0,255.0,-1.0,1.0);
    return image;
    
}
function showSelected(){
    var x = document.getElementById("style-name");
    for (var i = 0; i < x.length; i++) {
        if (x.options[i].value ==x.value){
            document.getElementById('style-'+x.options[i].value).className='d-block';
        }
        else{document.getElementById('style-'+x.options[i].value).className='d-none'}
    }
}
export async function bindPage() {
    toggleLoadingUI(true);
    const net = await posenet.load({
        architecture: guiState.input.architecture,
        outputStride: guiState.input.outputStride,
        inputResolution: guiState.input.inputResolution,
        multiplier: guiState.input.multiplier,
        quantBytes: guiState.input.quantBytes
    });


    //const STYLE_URL = 'https://eric-yim.github.io/style_encoder_ema_tfjs/model.json';
    //const GENERATOR_URL = 'https://eric-yim.github.io/generator_ema_tfjs/model.json';
    const STYLE_GENERATOR_URL = 'https://eric-yim.github.io/style_generator_merged_tfjs/model.json';
    // For Keras use tf.loadLayersModel().
    //const styleEncoder = await tfconv.loadGraphModel(STYLE_URL);
    //const generator = await tfconv.loadGraphModel(GENERATOR_URL);
    const styleGenerator = await tfconv.loadGraphModel(STYLE_GENERATOR_URL);
    toggleLoadingUI(false); 
    

    console.log("Models Loaded");
    let video;

    try {
        video = await loadVideo();
    } catch (e) {
        let info = document.getElementById('info');
        info.textContent = 'this browser does not support video capture,' +
            'or this device does not have a camera';
        info.style.display = 'block';
        throw e;
    }

    
    
    detectPoseInRealTime(video, net,styleGenerator);
}
bindPage();
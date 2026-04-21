const URL = "https://teachablemachine.withgoogle.com/models/rmUhEqnmd/";

let model, webcam, ctx, labelContainer, maxPredictions;

async function init() {
    const modelURL = URL + "model.json";
    const metadataURL = URL + "metadata.json";

    model = await tmPose.load(modelURL, metadataURL);
    maxPredictions = model.getTotalClasses();

    const size = 300;
    const flip = true;
    webcam = new tmPose.Webcam(size, size, flip);
    await webcam.setup();
    await webcam.play();

    window.requestAnimationFrame(loop);

    const canvas = document.getElementById("canvas");
    canvas.width = size;
    canvas.height = size;
    ctx = canvas.getContext("2d");

    labelContainer = document.getElementById("label-container");
    labelContainer.innerHTML = "";
    for (let i = 0; i < maxPredictions; i++) {
        labelContainer.appendChild(document.createElement("div"));
    }
}

async function loop(timestamp) {
    webcam.update();
    await predict();
    window.requestAnimationFrame(loop);
}

async function predict() {
    const { pose, posenetOutput } = await model.estimatePose(webcam.canvas);
    const prediction = await model.predict(posenetOutput);

    let highestClass = "";
    let highestProb = 0;

    for (let i = 0; i < prediction.length; i++) {
        const className = prediction[i].className;
        const probability = prediction[i].probability.toFixed(2);

        labelContainer.childNodes[i].innerHTML = `${className}: ${probability}`;

        if (probability > highestProb) {
            highestProb = probability;
            highestClass = className;
        }
    }

    const feedback = document.getElementById("feedback");

    if (highestClass === "Still and Calm") {
        feedback.innerHTML = "🟢 You are calm.";
    } else if (highestClass === "Mildly distressed") {
        feedback.innerHTML = "🟡 You are Mildly distressed -Try a small break.";
    } else if (highestClass === "Distressed and overwhelmed") {
        feedback.innerHTML = "🔴 Calm down and take deep breaths.";
    }

    drawPose(pose);
}

function drawPose(pose) {
    if (webcam.canvas) {
        ctx.drawImage(webcam.canvas, 0, 0);

        if (pose) {
            const minPartConfidence = 0.5;
            tmPose.drawKeypoints(pose.keypoints, minPartConfidence, ctx);
            tmPose.drawSkeleton(pose.keypoints, minPartConfidence, ctx);
        }
    }
}

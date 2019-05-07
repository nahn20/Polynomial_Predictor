//https://blog.tristansokol.com/2018/06/23/my-first-tensorflow.js-project/

//Make show fitness of the prediction
var viewArea = {
    x : -18,
    y : -8,
    sizeMultiplier : 35,
    load : function(){
        this.canvas = document.getElementById("canvas");
        this.ctx = this.canvas.getContext("2d");
        updateGuessCoordinates();
        setInterval(function(){draw();}, 20);
    }
}

var trainedCount = 0;
var polynomialDegree = 3;
var estimatePolynomialDegree = 3;
var learningRate = .000015;
var trainCoords = [[], []];
var guessCoords = [[], []];
var trainParamArray = [];
var paramArray = [];
updateLRFromDegree(estimatePolynomialDegree);
var optimizer = tf.train.sgd(learningRate); //Uses Stochastic Gradient Descent to optimize (https://developers.google.com/machine-learning/crash-course/reducing-loss/gradient-descent)
zeroParameters();
generatePoints();
function updateLRFromDegree(pd){
    switch(pd){
        case 1:
            learningRate = 0.09;
            break;
        case 2:
            learningRate = 0.007;
            break;
        case 3:
            learningRate = 0.0004;
            break;
        case 4:
            learningRate = 0.00002;
            break;
        case 5:
            learningRate = 0.000001;
            break;
    }
}
function generatePoints(){
    trainParamArray = [];
    trainCoords = [[], []];
    for(let i = 0; i < polynomialDegree+1; i++){
        trainParamArray[i] = (Math.random()-0.5);
    }
    let q = polynomialDegree*20;
    let varianceMultiplier = 2;
    if(document.getElementById("varianceMultiplier")){
        varianceMultiplier = document.getElementById("varianceMultiplier").value;
    }
    for(let i = 0; i < q; i++){
        trainCoords[0][i] = (i-q/2)/(q/10); //higher degree polynomials need more points
        trainCoords[1][i] = varianceMultiplier*(Math.random()-0.5); //Variance factor
        for(let q = 0; q < trainParamArray.length; q++){
            trainCoords[1][i] += trainParamArray[q]*Math.pow(trainCoords[0][i], q);
        }
    }
}
function generatePointsFromDataset(){
    trainCoords = [[], []];
    let xSet = document.getElementById("datasetX").value.split(",");
    let ySet = document.getElementById("datasetY").value.split(",");
    let x = [];
    let y = [];
    if(xSet.length == ySet.length){
        for(let i = 0; i < xSet.length; i++){
            trainCoords[0][i] = parseInt(xSet[i]);
            trainCoords[1][i] = parseInt(ySet[i]);
        }
    }
    else{
        alert("Error: Coordinate Length Mismatch!");
    }
}

function zeroParameters(){
    paramArray = [];
    for(let i = 0; i < estimatePolynomialDegree+1; i++){
        paramArray[i] = tf.variable(tf.scalar(0));
    }
}


function predict(x){
    return tf.tidy(function(){
        let sum = paramArray[0];
        for(let i = paramArray.length-1; i > 0; i--){
            sum = sum.add(paramArray[i].mul(tf.pow(x, i)));
        }
        return sum;
    });
}
async function updateGuessCoordinates(){
    let q = estimatePolynomialDegree*20;
    for(let i = 0; i < q; i++){
        guessCoords[0][i] = (i-q/2)/(q/20);
    }
    guessCoords[1] = await predict(guessCoords[0]).data();
}
function draw(){
    viewArea.ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawPoints(trainCoords[0], trainCoords[1], "black");
    drawPoints(guessCoords[0], guessCoords[1], "red")

    viewArea.ctx.fillStyle = "black";
    viewArea.ctx.fillText("Trained " + trainedCount + " times.", 10, canvas.height-20*(paramArray.length+1)-10);
    viewArea.ctx.fillText("Predicted Parameters:", 10, canvas.height-20*(paramArray.length)-10);
    for(let i = 0; i < paramArray.length; i++){
        viewArea.ctx.fillText((Math.round(10000000*paramArray[i].dataSync()[0])/10000000) + "x^" + i, 10, canvas.height-20*i-10);
    }

    viewArea.ctx.fillText("Actual Parameters:", canvas.width-100, canvas.height-20*(paramArray.length)-10);
    for(let i = 0; i < trainParamArray.length; i++){
        viewArea.ctx.fillText((Math.round(10000000*trainParamArray[i])/10000000) + "x^" + i, canvas.width-100, canvas.height-20*i-10);
    }
}
function drawPoints(x, y, color){
    for(var i = 0; i < x.length; i++){
        let side = viewArea.sizeMultiplier/8;
        viewArea.ctx.fillStyle = color;
        viewArea.ctx.fillRect(viewArea.sizeMultiplier*(x[i]-viewArea.x)-side/2, viewArea.sizeMultiplier*(y[i]-viewArea.y)-side/2, side, side);
    }
}
function goodness(prediction, actual){
    const error = prediction.sub(actual).square().mean();
    return error;
}
function train(){
    trainedCount++;
    optimizer.minimize(function(){
        const predsYs = predict(tf.tensor1d(trainCoords[0]));
        stepLoss = goodness(predsYs, tf.tensor1d(trainCoords[1]))
        if(document.getElementById("doStepLoss").checked){
            stepLoss.print();
        }

        return stepLoss;
    });
}
function runTest(){
    let count = document.getElementById("numTests").value;
    let timeDelay = document.getElementById("delayLength").value;
    for(let i = 0; i < count; i++){
        if(timeDelay == 0){
            train();
        }
        else{
            setTimeout(function(){train(); updateGuessCoordinates();}, i*timeDelay);
        }
    }
    setTimeout(function(){updateGuessCoordinates();}, count*timeDelay+10);
}
function reset(){
    trainedCount = 0;
    resetPD();
    updateLRFromDegree(estimatePolynomialDegree);
    resetLR();
    if(document.getElementById("newRandomPoints").checked){
        generatePoints();
    }
    else if(document.getElementById("newDataset").checked){
        generatePointsFromDataset();
    }
    zeroParameters();
    updateGuessCoordinates();

}
function resetPD(){
    let placeholderPD = parseInt(document.getElementById("degree").value);
    if(placeholderPD != 0){
        polynomialDegree = placeholderPD;
    }
    let estimatePlaceholderPD = parseInt(document.getElementById("evaluationDegree").value);
    if(estimatePlaceholderPD != 0){
        estimatePolynomialDegree = estimatePlaceholderPD;
    }
}
function resetLR(){
    let placeholderLR = document.getElementById("learningRate").value;
    if(placeholderLR != 0){
        learningRate = placeholderLR;
    }
    optimizer = tf.train.sgd(learningRate);
}
function updateUI(handler){
    if(document.getElementById("newDataset").checked && handler == 2){
        document.getElementById("newRandomPoints").checked = false;
    }
    if(document.getElementById("newRandomPoints").checked && handler == 1){
        document.getElementById("newDataset").checked = false;
    }

    if(document.getElementById("newRandomPoints").checked){
        document.getElementById("generationElements").style.opacity = 1;
    }
    else{
        document.getElementById("generationElements").style.opacity = 0;
    }

    if(document.getElementById("newDataset").checked){
        document.getElementById("datasetElements").style.opacity = 1;
    }
    else{
        document.getElementById("datasetElements").style.opacity = 0;
    }
}
document.addEventListener('keydown', function(event) {
    switch(event.keyCode){
        case 38:
            viewArea.y += 0.5;
            break;
        case 40:
            viewArea.y -= 0.5;
            break;
        case 37:
            viewArea.x += 0.5;
            break;
        case 39:
            viewArea.x -= 0.5;
            break;
        case 191:
            viewArea.sizeMultiplier -= 1;
            viewArea.x -= 20/viewArea.sizeMultiplier;
            break;
        case 16:
            viewArea.sizeMultiplier += 1;
            viewArea.x += 20/viewArea.sizeMultiplier;
            break;
        case 82:
            reset();
            break;
    }
    if(event.keyCode == 38 || event.keyCode == 37 || event.keyCode == 40 || event.keyCode == 39){
        event.preventDefault();
    }
    updateGuessCoordinates();
});
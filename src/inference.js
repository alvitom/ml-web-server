const tf = require("@tensorflow/tfjs-node");

const loadModel = async () => {
  const model = await tf.loadLayersModel("file://models/model.json");
  return model;
};

const predict = async (model, image) => {
  const tensor = tf.node.decodeJpeg(image).resizeNearestNeighbor([150, 150]).expandDims().toFloat();

  return model.predict(tensor).data();
};

module.exports = { loadModel, predict };

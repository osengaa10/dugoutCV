YoloDetectionNetwork
====================

Yolo detection network node is very similar to :ref:`NeuralNetwork` (in fact it extends it). The only difference is that this node
is specifically for the **(tiny) Yolo V3/V4** NN and it decodes the result of the NN on device. This means that :code:`Out` of this node is not a
:ref:`NNData` (a byte array) but a :ref:`ImgDetections` that can easily be used in your code.

How to place it
###############

.. tabs::

  .. code-tab:: py

    pipeline = dai.Pipeline()
    yoloDet = pipeline.createYoloDetectionNetwork()

  .. code-tab:: c++

    dai::Pipeline pipeline;
    auto yoloDet = pipeline.create<dai::node::YoloDetectionNetwork>();


Inputs and Outputs
##################

.. code-block::

              ┌───────────────────┐
              │                   │       out
              │                   ├───────────►
              │     Yolo          │
              │     Detection     │
  input       │     Network       │ passthrough
  ───────────►│-------------------├───────────►
              │                   │
              └───────────────────┘

**Message types**

- :code:`input` - :ref:`ImgFrame`
- :code:`out` - :ref:`ImgDetections`
- :code:`passthrough` - :ref:`ImgFrame`

Usage
#####

.. tabs::

  .. code-tab:: py

    pipeline = dai.Pipeline()
    yoloDet = pipeline.createYoloDetectionNetwork()
    yoloDet.setBlobPath(nnBlobPath)

    # Yolo specific parameters
    yoloDet.setConfidenceThreshold(0.5)
    yoloDet.setNumClasses(80)
    yoloDet.setCoordinateSize(4)
    yoloDet.setAnchors(np.array([10,14, 23,27, 37,58, 81,82, 135,169, 344,319]))
    yoloDet.setAnchorMasks({"side26": np.array([1, 2, 3]), "side13": np.array([3, 4, 5])})
    yoloDet.setIouThreshold(0.5)

  .. code-tab:: c++

    dai::Pipeline pipeline;
    auto yoloDet = pipeline.create<dai::node::YoloDetectionNetwork>();
    yoloDet->setBlobPath(nnBlobPath);

    // yolo specific parameters
    yoloDet->setConfidenceThreshold(0.5f);
    yoloDet->setNumClasses(80);
    yoloDet->setCoordinateSize(4);
    yoloDet->setAnchors({10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319});
    yoloDet->setAnchorMasks({{"side13", {3, 4, 5}}, {"side26", {1, 2, 3}}});
    yoloDet->setIouThreshold(0.5f);

Examples of functionality
#########################

- :ref:`22.1 - RGB & TinyYoloV3 decoding on device`
- :ref:`22.2 - RGB & TinyYoloV4 decoding on device`

Reference
#########

.. tabs::

  .. tab:: Python

    .. autoclass:: depthai.YoloDetectionNetwork
      :members:
      :inherited-members:
      :noindex:

  .. tab:: C++

    .. doxygenclass:: dai::node::YoloDetectionNetwork
      :project: depthai-core
      :members:
      :private-members:
      :undoc-members:

.. include::  ../../includes/footer-short.rst

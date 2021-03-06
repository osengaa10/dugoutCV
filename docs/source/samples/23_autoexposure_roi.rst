23 - Auto Exposure on ROI
=========================

This example shows how to dynamically set the Auto Exposure (AE) of the RGB camera dynamically, during application runtime,
based on bounding box position

Demo
####

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/aTqUwNL_9Bo" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>

Setup
#####

.. include::  /includes/install_from_pypi.rst

This example also requires MobilenetSDD blob (:code:`mobilenet-ssd_openvino_2021.2_5shave.blob` file) to work - you can download it from
`here <https://artifacts.luxonis.com/artifactory/luxonis-depthai-data-local/network/mobilenet-ssd_openvino_2021.2_5shave.blob>`__

Usage
#####

By default, AutoExposure region is adjusted based on neural network output. If desired, the region can be set manually.
You can do so by pressing one of the following buttons:

- `w` - move AE region up
- `s` - move AE region down
- `a` - move AE region left
- `d` - move AE region right
- `n` - deactivate manual region (switch back to nn-based roi)

Source code
###########

Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/23_autoexposure_roi.py>`__

.. literalinclude:: ../../../examples/23_autoexposure_roi.py
   :language: python
   :linenos:

.. include::  /includes/footer-short.rst

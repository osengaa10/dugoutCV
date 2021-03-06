CameraControl
=============

This message is used for controlling the :ref:`color camera <ColorCamera>` as well as the :ref:`mono camera <MonoCamera>`.
The message handles things like capturing still images, confifguring auto focus, anti banding, white balance,
scenes, effects etc.

Examples of functionality
#########################

- :ref:`14.1 - Color Camera Control`
- :ref:`19 - Mono Camera Control`
- :ref:`23 - Auto Exposure on ROI`

Reference
#########

.. tabs::

  .. tab:: Python

    .. autoclass:: depthai.CameraControl
      :members:
      :inherited-members:
      :noindex:

  .. tab:: C++

    .. doxygenclass:: dai::CameraControl
      :project: depthai-core
      :members:
      :private-members:
      :undoc-members:

.. include::  ../../includes/footer-short.rst

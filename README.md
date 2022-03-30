# VNOS
This is a project for the VNOS (Embedded Systems) class. It consists of a web application which works with a smart camera in the form of a raspberry pi. The camera captures images of a parking space (or spaces) and using a neural network model decides whether the space is occupied or empty. In the event of a change of occupancy status the application sends the user a notification or an email.

The project is coded using Python with the Flask library as a base. The neural network model is based on mAlexNet model which is based on this research [1]. It was converted from a Caffe model to a Keras model using https://github.com/microsoft/MMdnn.

[1] S. Rahman, M. Ramli, F. Arnia, A. Sembiring and R. Muharar, "Convolutional Neural Network Customization for Parking Occupancy Detection," 2020 International Conference on Electrical Engineering and Informatics (ICELTICs), 2020, pp. 1-6, doi: 10.1109/ICELTICs50595.2020.9315509.

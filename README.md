# bcd-FlaskPyWebIO
An end-to-end implementation of Breast Cancer Detection using [prosemble ML package](https://github.com/naotoo1/prosemble) within the Flask framework integrated in PyWebIO with deployment on Heroku platform as a service cloud.

## How to use
To diagnose breast cancer disease and return the confidence of the diagnosis,
1. click on the bcd-FlaskPyWebIO on the environments section and then click on view deployment or simply use the link https://bcd-flaskpywebio.herokuapp.com/
2. After opening the link, enter the value for Radius_mean  and click on ```submit ``` to proceed or ```reset``` to enter new value
3. Enter the value for Radius_texture and and click on ```submit ``` to proceed or ```reset``` to enter new value
4. Select the method to proceed and  click on ```submit```

### FastAPI framework Version
For fastapi framework deployment version refer to [bcd-fastapi](https://github.com/naotoo1/bcd-fastapi)

### FlaskFlasgger and Streamlit framework Version
For Flask framework with Flasgger as well as Streamlit version: [bcd-flaskflasgger](https://github.com/naotoo1/bcd-flaskflasgger)

### Advance Inclusions
For advanced breast cancer diagnosis, utilizing a multiple reject classification strategy for improving the reliability of diagnosis, the  class-related confidence thresh-holds determined by the implementation of the CRT algorithm where users want low rejection rate and high reliability has been shown empirically in 
[Multiple Reject Classification Strategy](https://github.com/naotoo1/Multiple-Reject-Classification-Options)

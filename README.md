# communication-som
A simple simulation of languages evolving. 

## Instructions
To run the code, the only thing that is needed is that you have the pip packages in the "requirements.txt" file, install these with `pip install -r requirements.txt`. The "main.py" file can be run once the virtual environment is active. There are no command line options, but the constants at the top of the `main()` function can be set to different values for different results. After each run, the grid will display and the results will be saved to the end of a line in the "results.csv" file. 

### Self-Talk
The variable `USE_SELF_TALK` can be set to true or false. If this is set to true, self-talk will be active. 

### Model Type
The variable `SOCIAL_ENTITY_TYPE` can be set to `'standard'` or `'invertible'`. In the latter case, the Moore-Penrose Pseudoinverse-based model is used for encoding and decoding messages. 

### Synthetic Data
For faster processing and to allow for smaller model sizes, an option was added to run the model with data that is simulated. This can be adjusted with the variable `USE_SYNTHETIC_DATA`. By default, it is set to true. 

### Calculating P-Values
The p-values mentioned in the paper were calculated by running the "welch_t_test.py" file. This calculates values based on the data in the "results.csv" file.

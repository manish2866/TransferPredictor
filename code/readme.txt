Instructions to use the web app
1. Open the terminal window
2. Run pip3 install streamlit to install streamlit
3. Use the code streamlit run app.py
4. Upon running this code, the web app is opened in the default browser
5. Using the upload csv button, user can upload a clean csv file which contains the following fields and also should adhere to the datatypes
['name', 'full_name', 'birth_date', 'age', 'height_cm', 'weight_kgs',
              'positions', 'nationality', 'overall_rating', 'potential', 'value_euro',
              'wage_euro', 'preferred_foot', 'international_reputation(1-5)',
              'weak_foot(1-5)', 'skill_moves(1-5)', 'body_type',
              'release_clause_euro', 'crossing', 'finishing', 'heading_accuracy',
              'short_passing', 'volleys', 'dribbling', 'curve', 'freekick_accuracy',
              'long_passing', 'ball_control', 'acceleration', 'sprint_speed',
              'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina',
              'strength', 'long_shots', 'aggression', 'interceptions', 'positioning',
              'vision', 'penalties', 'composure', 'marking', 'standing_tackle',
              'sliding_tackle']
6. The code is written in such a way that, it takes the input dataset performs feature engineering, splits the dataset, trains and tests the dataset after fitting the model.
7. It also provides certain visualizations. 
8. The user can input his own data fields and predict based on the previously trained model.

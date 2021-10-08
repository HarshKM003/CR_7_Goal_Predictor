from flask import Flask,render_template,request
import pickle
import numpy as np

model = pickle.load(open('D:/ML Projects/CR_goal/saved_model/model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def cr_goal_predictor():
    print('hello user')
    match_id = float(request.form.get('match_event_id'))
    loc_x =  float(request.form.get('location_x'))
    loc_y =  float(request.form.get('location_y'))
    time_rem= float(request.form.get('remaining_min'))
    power_shot =  float(request.form.get('power_of_shot'))
    knockout_mat =  float(request.form.get('knockout_match'))
    season= float(request.form.get('game_season'))
    sec_rem =  float(request.form.get('remaining_sec'))
    dis_of_shot =  float(request.form.get('distance_of_shot'))
    area_of_shot =  float(request.form.get('area_of_shot'))
    shot_b = float(request.form.get('shot_basics'))
    range= float(request.form.get('range_of_shot'))
    opponent = float(request.form.get('team_name'))
    home_adv = float(request.form.get('home / away'))
    shot_id = float(request.form.get('shot_id_number'))
    loc = float(request.form.get('lat / lng'))
    shot_type = float(request.form.get('type_of_shot'))
    comb_shot = float(request.form.get('type_of_combined_shot'))
    matchid = float(request.form.get('match_id'))
    teamid = float(request.form.get('team_id'))

    result = model.predict(np.array([match_id,loc_x,loc_y,time_rem,power_shot,knockout_mat,season,sec_rem,
                                     dis_of_shot,area_of_shot,shot_b,range,opponent,home_adv,shot_id,loc,shot_type,
                                     comb_shot,matchid,teamid]))
    if result[0] == 0:
        result = 'It will not be a goal!!!!'
    if result[0] == 1:
        result = 'It will be a goal!!!!'

    return render_template('index.html', result= result)


if __name__ == '__main__':
    app.run(debug=True)

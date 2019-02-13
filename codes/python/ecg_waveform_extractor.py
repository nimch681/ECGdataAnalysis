
class rr_intervals:
    def __init__(self):
        self.pre_rr_interval = 0
        self.post_rr_interval = 0 
        

def pre_pos_rr_interval(record):
    pre_r = record.segmented_R_pos[0]
    current_r = 0
    post_r = 0 
    pre_rr_interval = []
    post_rr_interval = []
    
    for r in range(0, len(record.segmented_R_pos)):
        current_r = record.segmented_R_pos[r]
        if(r < len(record.segmented_R_pos)-1):
            post_r = record.segmented_R_pos[r+1]
        else:
            post_r = current_r
        pre_rr_interval.append(record.time[current_r] - record.time[pre_r])
        post_rr_interval.append(record.time[post_r] - record.time[current_r] )
        pre_r = current_r
        
        
    return  pre_rr_interval, post_rr_interval


def rr_average_sample(pre_rr_interval,ten=True,fifty=False, all=True):
     
    rr_ten = []
    rr_fifty = []
    rr_all = []
    num_max_beat = len(pre_rr_interval)
    if all == True:
        rr_average = average(pre_rr_interval)
        rr_all = [rr_average] * num_max_beat

    for i in range(0,num_max_beat):
        rr_average_10 = 0
        rr_average_50 = 0
        if ten == True:
            if i > num_max_beat-6:
                rr_average_10 = average(pre_rr_interval[i-5:num_max_beat])
                print(i)
            if i < 5:
                rr_average_10 = average(pre_rr_interval[0:i+5])
                print(i)

            if(i > 5 and i < num_max_beat-6):
                rr_average_10 = average(pre_rr_interval[i-5:i+5])
            
            rr_ten.append(rr_average_10)
                
        if fifty == True:
            
            if i > num_max_beat-26:
                rr_average_50 = average(pre_rr_interval[i-25:num_max_beat])

            if i < 25:
                rr_average_50 = average(pre_rr_interval[0:i+25])

            if(i > 25 and i < num_max_beat-26):
                rr_average_50 = average(pre_rr_interval[i-25:i+25])
            
            rr_fifty.append(rr_average_50)

        
    return rr_ten, rr_fifty, rr_all

def average(numbers):
    return float(sum(numbers)) / len(numbers)

pre, pos = pre_pos_rr_interval(mit100)

new_list = list(range(len(pre)))

average_10 = rr_average_sample(pre)

    




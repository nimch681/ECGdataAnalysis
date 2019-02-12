
class waveform_extractor:
    def __init__():
        

def rr_interval(record):
    pre_r = record.segmented_R_pos[0]
    current_r = 0
    post_r = 0 
    pre_rr_interval = []
    post_rr_interval = []
    for r in range(0, len(record.segmented_R_pos)):
        current_r = record.segmented_R_pos[r]
       # if(r < len(record.segmented_R_pos)):
        #    post_r = record.segmented_R_pos[r+1]
       # else:
          #  post_r = current_r
        #pre_rr_interval.append(record.time[current_r] - record.time[pre_r])
        #post_rr_interval.append(record.time[post_r] - record.time[current_r])
        #pre_r = current_r
        
    return pre_rr_interval, post_rr_interval

pre, pos = rr_interval(mit100)
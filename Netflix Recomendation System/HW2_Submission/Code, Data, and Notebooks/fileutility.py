
def extract_num_movies(filepath):
    with open(filepath) as f:
        num_lines = sum(1 for line in f)
        return num_lines

def extract_user_dict(filepath):
    user_dict = {}
    user_count = 0
    with open(filepath) as f:
        for line in f:
            id = int(line.split(',')[1])
            if id not in user_dict:
                user_dict[id] = user_count
                user_count+=1
    return user_count, user_dict

def get_num_lines(filepath):
    return extract_num_movies(filepath)

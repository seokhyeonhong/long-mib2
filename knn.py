import torch
import pdb
import numpy as np
from annoy import *     # https://github.com/spotify/annoy 

# Set options for constructing tree
OPTIONS = {
    'pose_dim': 132,
    'traj_dim': 4,
    'traj_len': 15,
    'traj_pos_w': 10,
    'traj_dir_w': 2,
    'pose0_w': 1,
}


def is_same_motion(motion_id, first_idx, last_idx):
    if motion_id[first_idx] == motion_id[last_idx]:
        return True
    return False


def get_indices(motion, motion_id, trj_len):
    pidx2nidx = []
    nidx2pidx = []

    # pidx becomes the first pose for each data sample
    pidx_n = motion.size(0)
    for pidx in range(pidx_n):
        is_not_end = (pidx + trj_len) < pidx_n

        if is_not_end and is_same_motion(motion_id, pidx, pidx + trj_len):
            # Save index for pose and data sample
            pidx2nidx.append(len(nidx2pidx))        # 이번 pose는 몇 번째 data sample에 포함되는지 저장
            nidx2pidx.append(pidx)                  # 이번 data sample은 몇 번째 pose 인지 저장
        else:
            pidx2nidx.append(-1)

    return pidx2nidx, nidx2pidx


def get_trj_feature(motion, motion_id, pidx, options):
    _, _, traj_pos, traj_dir = torch.split(motion, [options['pose_dim'], 3, 2, 2], dim=-1)
    pidx_trj_pos = traj_pos[pidx] * options['traj_pos_w']
    pidx_trj_dir = traj_dir[pidx] * options['traj_dir_w']
    f_trj = torch.cat([pidx_trj_pos, pidx_trj_dir]) # [4]
    
    for i in range(1, options['traj_len']):
        assert(is_same_motion(motion_id, pidx, pidx + i))
        weighted_traj_pos = traj_pos[pidx + i] * options['traj_pos_w']
        weighted_traj_dir = traj_dir[pidx + i] * options['traj_dir_w']
        f_trj = torch.cat([f_trj, weighted_traj_pos, weighted_traj_dir])
    return f_trj


def get_pose_feature(motion, pidx, options):
    local_R6, _, _, _ = torch.split(motion, [options['pose_dim'], 3, 2, 2], dim=-1)
    f_pose = local_R6[pidx] * options['pose0_w']
    return f_pose


def get_search_tree(motion, motion_id, nidx2pidx, options):
    # Initialize search tree
    feature_dim = options['pose_dim'] + options['traj_len'] * options['traj_dim']
    tree = AnnoyIndex(feature_dim, 'euclidean')

    # Obtain feature vector for each clip
    pnt_n = len(nidx2pidx)
    print(f'Initializing search tree with {pnt_n} samples...')

    for i in range(pnt_n): 
        pidx = nidx2pidx[i]
        f_pose0 = get_pose_feature(motion, pidx, options)              # [132]
        f_trj = get_trj_feature(motion, motion_id, pidx, options)      # [60]
        data_sample = torch.cat([f_pose0, f_trj])                      # [192]
        
        # Add to search tree
        tree.add_item(i, data_sample)
        print(f'Added {i}th data to search tree', end='\r')

    tree.build(1)
    print()
    return tree


def get_query(q_motion, options):
    assert(q_motion.size(0) == options['traj_len'])     # [15, feature]
    q_motion_id = torch.zeros([q_motion.size(0)])       # [15]
    index = torch.tensor(range(options['traj_len']))
    q_motion_id.index_fill_(0, index, 0)                # all poses included in a same motion (fill with same integer)

    f_pose0 = get_pose_feature(q_motion, 0, options)   
    f_trj = get_trj_feature(q_motion, q_motion_id, 0, options)      
    query = torch.cat([f_pose0, f_trj])
    return query       


def main():
    total_motion = torch.from_numpy(
    np.load("./data/test/motion_length101_offset20_fps30.npy"))
    B, T, D = total_motion.shape

    # ! test on 100 samples -> leads to 8600 samples for traj_len = 15
    num_sample = 100
    test_motion = total_motion[:100]

    # Annotate index of motions that each pose is included
    motion_id = torch.zeros([num_sample, T])  # [4432, 101]
    index = torch.tensor(range(T))
    for i in range(num_sample):
        motion_id[i].index_fill_(0, index, i)

    # * Construct search tree
    # Make all motions into one long sequence
    motion_list = torch.reshape(test_motion, [-1, D])  # [447632, 139]
    motion_id = torch.reshape(motion_id, [-1])  # [447632]

    # Get mapping indices 
    m_pidx2nidx, m_nidx2pidx = get_indices(motion_list, motion_id, OPTIONS['traj_len'])

    search_tree = get_search_tree(motion_list, motion_id, m_nidx2pidx, OPTIONS)
    tree_name = 'test.ann'
    search_tree.save(tree_name)
    print(f'Saved search tree as {tree_name}!\n')

    # * Search for motion clip
    # Load saved tree
    feature_dim = OPTIONS['pose_dim'] + OPTIONS['traj_len'] * OPTIONS['traj_dim']
    loaded_tree = AnnoyIndex(feature_dim, 'euclidean')
    loaded_tree.load(tree_name)
    print(f'Loaded search tree {tree_name}!')

    # Query for k similar motions
    k = 10
    q_motion = total_motion[100][:OPTIONS['traj_len']]  # [15, 139]
    query = get_query(q_motion, OPTIONS)
    results = loaded_tree.get_nns_by_vector(query, k, -1, True)   # returns indices of k nearest neighbor
    results_idx = results[0]
    results_dist = results[1]
    # print(loaded_tree.get_n_items())
    # print(results)

    # Retrieve full motion based on pidx
    neighbor_motions = torch.zeros([k, OPTIONS['traj_len'], D])    # [k, traj_len, D]
    for i in range(len(results_idx)):
        searched_pidx = m_nidx2pidx[results_idx[i]]
        searched_motion = motion_list[searched_pidx:searched_pidx + OPTIONS['traj_len']]
        neighbor_motions[i] = searched_motion
    print(neighbor_motions.shape)

    # Checking if the searched motion is actually similar
    # print(q_motion[0][:6])
    # for i in range(k):
    #     print(neighbor_motions[i][0][:6])
    #     print(results_dist[i])


if __name__ == "__main__":
    main()
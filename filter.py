def filter_one(lines):
    rho_threshold = 15
    theta_threshold = 0.1

    # how many lines are similar to a given one
    similar_lines = {i: [] for i in range(len(lines))}
    for i in range(len(lines)):
        for j in range(len(lines)):
            if i == j:
                continue

            rho_i, theta_i = lines[i][0]
            rho_j, theta_j = lines[j][0]
            if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                similar_lines[i].append(j)

    # ordering the INDECES of the lines by how many are similar to them
    indices = [i for i in range(len(lines))]
    indices.sort(key=lambda x: len(similar_lines[x]))

    # line flags is the base for the filtering
    line_flags = len(lines)*[True]
    for i in range(len(lines) - 1):
        # if we already disregarded the ith element in the ordered list then we don't care (we will not delete anything based on it and we will never reconsider using this line again)
        if not line_flags[indices[i]]:
            continue

        # we are only considering those elements that had less similar line
        for j in range(i + 1, len(lines)):
            # and only if we have not disregarded them already
            if not line_flags[indices[j]]:
                continue

            rho_i, theta_i = lines[indices[i]][0]
            rho_j, theta_j = lines[indices[j]][0]
            if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                # if it is similar and have not been disregarded yet then drop it now
                line_flags[indices[j]] = False

    return line_flags


def filter_two(lines,line_flags):
    filtered_lines = []
    for i in range(len(lines)):  # filtering
        if line_flags[i]:
            filtered_lines.append(lines[i])

    return filtered_lines
import random

layouts = list(range(10))
styles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11]

layouts_and_styles = []
for l in layouts:
    for s in styles:
        layouts_and_styles.append([l, s])

num_envs = 5
sampled_split = random.sample(layouts_and_styles, num_envs)
print(sampled_split)

# env split of 5: [[3, 0], [0, 4], [4, 5], [5, 4], [6, 1]]
import numpy as np

class CategoryManager:
    def __init__(self, category_file='dataset/list_category_cloth.txt'):
        data = np.loadtxt(category_file, dtype=str, delimiter=" ", skiprows=2, usecols=(0, 1))
        line_numbers = np.arange(1, len(data) + 1)
        data = np.column_stack((data[:, 0], line_numbers))
        self.categories = {int(d[1]): d[0] for d in data}

    def get_category_name(self, topwear_category=None, bottomwear_category=None, both_category=None):
        """Return the category name for a given category ID."""
        if topwear_category:
            category_id = topwear_category + 1
        elif bottomwear_category:
            category_id = bottomwear_category + 21
        elif both_category:
            category_id = both_category + 37
        else:
            return "No input"
        return self.categories.get(category_id, None)

    def get_all_categories(self):
        """Return the entire dictionary of categories."""
        return self.categories

    def __len__(self):
        """Return the number of categories."""
        return len(self.categories)
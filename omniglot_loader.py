import os
import random
import numpy as np
import math


class OmniglotLoader:


    def __init__(self, dataset_path, batch_size):

        self.dataset_path = dataset_path
        self.train_dictionary = {}
        self.validation_dictionary = {}
        self.evaluation_dictionary = {}
        self.dimension = 128
        self.batch_size = batch_size

        self.load_dataset()


    def load_dataset(self):

        train_path = os.path.join(self.dataset_path, 'encodings.pickle')
        validation_path = os.path.join(self.dataset_path, 'ec.pickle')

        # First let's take care of the train alphabets
        with open(train_path, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')

        myset = list(set(dict["names"]))
        for j in range(len(myset)):
            l = []
            for i in range(len(dict["names"])):
                if dict["names"][i] == myset[j]:
                    f = np.asarray(dict["encodings"][i])
                    f = f[:,np.newaxis]
                    f = f.transpose()
                    l.append(f)
            self.train_dictionary[myset[j]] = l

        # Now it's time for the validation alphabets
        with open(validation_path, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')

        #validation data    
        myset = list(set(dict["names"]))
        for j in range(len(myset)):
            l = []
            for i in range(len(dict["names"])):
                if dict["names"][i] == myset[j]:
                    f = np.asarray(dict["encodings"][i])
                    f = f[:,np.newaxis]
                    f = f.transpose()
                    l.append(f)
            self.evaluation_dictionary[myset[j]] = l

    def split_train_datasets(self):
        """ Splits the train set in train and validation

        Divide the 30 train alphabets in train and validation with
        # a 80% - 20% split (24 vs 6 alphabets)

        """

        available_char = list(self.train_dictionary.keys())
        number_of_char = len(available_char)

        train_indexes = random.sample(
            range(0, number_of_char - 1), int(0.8 * number_of_char))

        # If we sort the indexes in reverse order we can pop them from the list
        # and don't care because the indexes do not change
        train_indexes.sort(reverse=True)

        for index in train_indexes:
            self._train_char.append(available_char[index])
            available_char.pop(index)

        for i in available_char:
            self.train_dictionary.pop(i, None)
        # The remaining alphabets are saved for validation
        self._validation_char = available_char

        for i in range(len(validation_char)):
                 self.validation_dictionary[validation_char[i]] = self.train_dictionary[validation_char[i]]

    def _convert_path_list_to_images_and_labels(self, path_list, is_one_shot_task):
 
        number_of_pairs = int(len(path_list) / 2)
        pairs_of_images = [np.zeros(
            (number_of_pairs,1, self.dimension)) for i in range(2)]
        labels = np.zeros((number_of_pairs, 1))

        for pair in range(number_of_pairs):
            image = path_list[pair * 2]
            image = np.asarray(image).astype(np.float64)
            pairs_of_images[0][pair, 1, :, 0] = image

            image = path_list[pair * 2 + 1]
            image = np.asarray(image).astype(np.float64)
            pairs_of_images[1][pair, :, :, 0] = image

            if not is_one_shot_task:
                if (pair + 1) % 2 == 0:
                    labels[pair] = 0
                else:
                    labels[pair] = 1

            else:
                if pair == 0:
                    labels[pair] = 1
                else:
                    labels[pair] = 0

        if not is_one_shot_task:
            random_permutation = np.random.permutation(number_of_pairs)
            labels = labels[random_permutation]
            pairs_of_images[0][:, :, :,
                               :] = pairs_of_images[0][random_permutation, :, :, :]
            pairs_of_images[1][:, :, :,
                               :] = pairs_of_images[1][random_permutation, :, :, :]

        return pairs_of_images, labels

    def get_train_batch(self):

        available_characters = list(self.train_dictionary.keys())
        number_of_characters = len(available_characters)

        batch_images_path = []

        # If the number of classes if less than self.batch_size/2 we have to repeat characters
        selected_characters_indexes = [random.randint(
            0, number_of_characters-1) for i in range(self.batch_size)]
        
        for index in selected_characters_indexes:
            current_character = available_characters[index]
            available_images = self.train_dictionary[current_character]
            
            # Random select a 3 indexes of images from the same character (Remember
            # that for each character we have 20 examples).

            image_indexes = random.sample(range(0, 15), 3)
            
            image = self.train_dictionary[current_character][0]
            batch_images_path.append(image)
            image = self.train_dictionary[current_character][1]
            batch_images_path.append(image)

            # Now let's take care of the pair of images from different characters
            image = self.train_dictionary[current_character][2]
            batch_images_path.append(image)
            different_characters = available_characters[:]
            different_characters.pop(index)
            different_character_index = random.sample(
                range(0, number_of_characters - 1), 1)
            current_character = different_characters[different_character_index[0]]
            available_images = self.train_dictionary[current_character]
            image_indexes = random.sample(range(0, 15), 1)
            image = self.train_dictionary[current_character][0]
            batch_images_path.append(image)

        images, labels = self._convert_path_list_to_images_and_labels(
            batch_images_path, is_one_shot_task=False)

        return images, labels

    def get_one_shot_batch(self, support_set_size, is_validation):
        # Set some variables that will be different for validation and evaluation sets
        if is_validation:
            dictionary = self.validation_dictionary
        else:
            
            dictionary = self.evaluation_dictionary

        available_characters = list(dictionary.keys())
        number_of_characters = len(available_characters)

        batch_images_path = []

        test_character_index = random.sample(
            range(0, number_of_characters), 1)

        # Get test image
        current_character = available_characters[test_character_index[0]]

        available_images = dictionary[current_character]

        image_indexes = random.sample(range(0, 15), 2)

        test_image = (dictionary[current_character])[image_indexes[0]]
        batch_images_path.append(test_image)
        image = (dictionary[current_character])[image_indexes[1]]
        batch_images_path.append(image)

        # Let's get our test image and a pair corresponding to
        if support_set_size == -1:
            number_of_support_characters = number_of_characters
        else:
            number_of_support_characters = support_set_size

        different_characters = available_characters[:]
        different_characters.pop(test_character_index[0])

        # There may be some alphabets with less than 20 characters
        if number_of_characters < number_of_support_characters:
            number_of_support_characters = number_of_characters

        support_characters_indexes = random.sample(
            range(0, number_of_characters - 1), number_of_support_characters - 1)

        for index in support_characters_indexes:
            current_character = different_characters[index]
            available_images = dictionary[current_character]
            image_indexes = random.sample(range(0, 15), 1)
            image = (dictionary[current_character])[image_indexes[0]]
            batch_images_path.append(test_image)
            batch_images_path.append(image)

        images, labels = self._convert_path_list_to_images_and_labels(
            batch_images_path, is_one_shot_task=True)

        return images, labels

    def one_shot_test(self, model, support_set_size, number_of_tasks_per_alphabet,
                      is_validation):

        if is_validation:

            print('\nMaking One Shot Task on validation alphabets:')
        else:

            print('\nMaking One Shot Task on evaluation alphabets:')

        
        mean_alphabet_accuracy = 0
        for _ in range(number_of_tasks_per_alphabet):
            images, _ = self.get_one_shot_batch(
                support_set_size, is_validation=is_validation)
            probabilities = model.predict_on_batch(images)

            # Added this condition because noticed that sometimes the outputs
            # of the classifier was almost the same in all images, meaning that
            # the argmax would be always by defenition 0.
            if np.argmax(probabilities) == 0 and probabilities.std()>0.01:
                accuracy = 1.0
            else:
                accuracy = 0.0

            mean_alphabet_accuracy += accuracy
        mean_alphabet_accuracy /= number_of_tasks_per_alphabet

        print('accuracy: ' + str(mean_alphabet_accuracy))

        return mean_alphabet_accuracy

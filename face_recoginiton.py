# -*- coding: utf-8 -*-
from __future__ import print_function
import click
import os
import re
import face_recognition.api as face_recognition
import multiprocessing
import itertools
import sys
import PIL.Image
import numpy as np

files_recognized = []
best_performance_recognition = {}

def scan_known_people(known_people_folder):
    known_names = []
    known_face_encodings = []
    train_dir = os.listdir(known_people_folder)
    for person in train_dir:
        if person == ".DS_Store":
            pass
        else:
            pix = os.listdir(known_people_folder + person)


        # Loop through each training image for the current person
            for person_img in pix:
                if person_img == ".DS_Store":
                    pass
                else:
                    # Get the face encodings for the face in each image file
                    face = face_recognition.load_image_file(known_people_folder + person + "/" + person_img)
                    face_bounding_boxes = face_recognition.face_locations(face)

                    #If training image contains exactly one face
                    if len(face_bounding_boxes) == 1:
                        face_enc = face_recognition.face_encodings(face)[0]
                        # Add face encoding for current image with corresponding label (name) to the training data
                        known_names.append(person)
                        known_face_encodings.append(face_enc)
                    else:
                        if len(face_bounding_boxes) == 0:
                            click.echo("WARNING: No faces found in {}. Ignoring file.".format(person+'/'+person_img))
                        else:
                            click.echo("WARNING: More than one face found in {}. Ignoring file.".format(person+'/'+person_img))
    return known_names, known_face_encodings

def print_result(filename, name, distance, show_distance=False):
    only_filename = filename.split("/")
    filename = only_filename[-1]
    if filename in files_recognized:
        if best_performance_recognition[filename] > distance:
            best_performance_recognition[filename] = distance
    else:
        if show_distance:
            best_performance_recognition[filename] = distance
            files_recognized.append(filename)
            print("{} -> {},{}".format(filename, name, distance))
        else:
            best_performance_recognition[filename] = distance
            files_recognized.append(filename)
            print("{} -> {}".format(filename, name))

def print_final_performances():
    for pair in best_performance_recognition.items():
        print('Best performance for '+str(pair[0]) + ' : ' + str(pair[1]))

def test_image(image_to_check, known_names, known_face_encodings, tolerance=0.6, show_distance=False):
    unknown_image = face_recognition.load_image_file(image_to_check)

    # Scale down image if it's giant so things run a little faster
    if max(unknown_image.shape) > 1600:
        pil_img = PIL.Image.fromarray(unknown_image)
        pil_img.thumbnail((1600, 1600), PIL.Image.LANCZOS)
        unknown_image = np.array(pil_img)

    unknown_encodings = face_recognition.face_encodings(unknown_image)

    for unknown_encoding in unknown_encodings:
        distances = face_recognition.face_distance(known_face_encodings, unknown_encoding)
        result = list(distances <= tolerance)

        if True in result:
            [print_result(image_to_check, name, distance, show_distance) for is_match, name, distance in zip(result, known_names, distances) if is_match]
        else:
            print_result(image_to_check, "Unknown person", None, show_distance)

    if not unknown_encodings:
        # print out fact that no faces were found in image
        print_result(image_to_check, "No person found", None, show_distance)


def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]


def process_images_in_process_pool(images_to_check, known_names, known_face_encodings, number_of_cpus, tolerance, show_distance):
    if number_of_cpus == -1:
        processes = None
    else:
        processes = number_of_cpus

    # macOS will crash due to a bug in libdispatch if you don't use 'forkserver'
    context = multiprocessing
    if "forkserver" in multiprocessing.get_all_start_methods():
        context = multiprocessing.get_context("forkserver")

    pool = context.Pool(processes=processes)

    function_parameters = zip(
        images_to_check,
        itertools.repeat(known_names),
        itertools.repeat(known_face_encodings),
        itertools.repeat(tolerance),
        itertools.repeat(show_distance)
    )

    pool.starmap(test_image, function_parameters)


@click.command()
@click.option('--known_people_folder', default='people/')
@click.option('--image_to_check', default='uk/')
@click.option('--cpus', default=1, help='number of CPU cores to use in parallel (can speed up processing lots of images). -1 means "use all in system"')
@click.option('--tolerance', default=0.6, help='Tolerance for face comparisons. Default is 0.6. Lower this if you get multiple matches for the same person.')
@click.option('--show-distance', default=False, type=bool, help='Output face distance. Useful for tweaking tolerance setting.')
@click.option('--show-final', default=False, type=bool, help='Show best result of recogintion. Use it to get an indicator of the accuracy of the results (lower is better).')
def main(known_people_folder, image_to_check, cpus, tolerance, show_distance, show_final):
    known_names, known_face_encodings = scan_known_people(known_people_folder)

    # Multi-core processing only supported on Python 3.4 or greater
    if (sys.version_info < (3, 4)) and cpus != 1:
        click.echo("WARNING: Multi-processing support requires Python 3.4 or greater. Falling back to single-threaded processing!")
        cpus = 1

    if os.path.isdir(image_to_check):
        if cpus == 1:
            [test_image(image_file, known_names, known_face_encodings, tolerance, show_distance) for image_file in image_files_in_folder(image_to_check)]
        else:
            process_images_in_process_pool(image_files_in_folder(image_to_check), known_names, known_face_encodings, cpus, tolerance, show_distance)
    else:
        test_image(image_to_check, known_names, known_face_encodings, tolerance, show_distance)
    if show_final:
        print_final_performances()


if __name__ == "__main__":
    main()

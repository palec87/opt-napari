# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:26:55 2024

@author: GiorgiaT
"""

    
        # def start_preprocessing(self):
        #     ''''
        #     GT: Creates an instance of the Processor. Needed when creating the plugin(I think)
        #     '''
        #     self.isCalibrated = False
            
        #     if hasattr(self, 'h'):
        #         self.stop_opt_processor()
        #         self.start_opt_processor()
        #     else:
        #         print('Reset')
        #         self.h = OPTProcessor() 

        # def stop_preprocessing(self):
        #     if hasattr(self, 'h'):
        #         delattr(self, 'h')

        # def reset_preprocessing(self,*args):
            
        #     self.isCalibrated = False
        #     self.stop_opt_processor()
        #     self.start_opt_processor() 
    
# @dataclass
# class Backtrack:
#     """ 
#     Supports one undo operation. Only for inplace operations.
#     Will break if:
#         You operate on more datasets in parallel
#     """
#     # def __init__(self) -> None:
#     raw_data: np.ndarray = None
#     last_data: np.ndarray = None
#     # roi_def: tuple = ()  # indices refer always to raw data
#     track: bool = False
#     inplace: bool = False
#     history_item = dict()

#     def update_tracking(self, value: bool):
        
#         self.track = value
#         if self.track is True:


#     def update_history(self, operation, data_dict):
#         if operation == 'roi':
#             pass
#         elif operation == 'bin':
#             pass
#         elif operation == 'correct':
#             pass
#         else:
#             raise TypeError('Unknown operation to update history.')
        
#         self.operation = operation
        
#     def revert_history(self):
#         pass

#     def revert_to_raw(self):
#         pass


# history = Backtrack()


# ROI
# @magic_factory(call_button="Select ROI")
# def select_ROIs(viewer: Viewer,
#                 image: Image,
#                 points_layer: Points,
#                 roi_height: Annotated[int, {'min': 1, 'max': 5000}] = 200,
#                 roi_width: Annotated[int, {'min': 1, 'max': 5000}] = 200,
#                 inplace: bool = True,
#                 track_history: bool = False,
#                 ):
#     shared_variables.track = track_history
#     original_stack = np.asarray(image.data)
#     points = np.asarray(points_layer.data)
#     notifications.show_info(f'UL corner coordinates: {points[0][1:].astype(int)}')

#     selected_roi, roi_pars = select_roi(original_stack,
#                                         points[-1],
#                                         roi_height,
#                                         roi_width)
#     # if I do this twice, I will get wrong indicis for corrections!!!
#     update_roi_pars(roi_pars)
#     # shared_variables.roi_def: roi_pars
#     # partial solution to tracking history
#     # TODO: does not work, if you are on wrong layer, or perhaps
#     # user deleted one layer going back one step, and wants to do another step
#     # Or changes the inplace parameter inbetween
#     # LOts of lose ends here!!!!
#     if inplace:
#         update_history(inplace, image, selected_roi)
#     else:
#         viewer.add_image(selected_roi)


# # this will not work, if user applies it once on the roi and then again on the
# # raw, fuck me...
# # Fixing means making the history a proper class.
# def update_roi_pars(roi_pars):
#     if shared_variables.roi_def == ():
#         shared_variables.roi_def = roi_pars
#     else:
#         i1, i2, i3, i4 = shared_variables.roi_def
#         j1, j2, j3, j4 = roi_pars
#         shared_variables.roi_def = (
#             i1 + j1, i1 + j1 + j2,
#             i3 + j3, i3 + j3 + j4,
#         )

# def update_history(inplace: bool, image: Image, new_image: np.ndarray):
#     if not shared_variables.track:
#         notifications.show_info('Tracking not enabled, no history to update (saving RAM).')
#         return
#     if not inplace:
#         notifications.show_info('Not an inplace operation, no tracking, just delete the layer.')
#         return

#     if shared_variables.raw_data is None:
#         shared_variables.raw_data = image.data.copy()
#     else:
#         shared_variables.last_data = image.data.copy()

#     image.data = new_image
#     return


# # Revert history
# @magic_factory(call_button="Undo Last")
# def revert_last(viewer: Viewer, image: Image, inplace: bool):
#     if not shared_variables.track:
#         notifications.show_info('Tracking not enabled, no history to update (saving RAM).')
#         return

#     if shared_variables.last_data is not None:
#         notifications.show_info('Reverting to last data')
#         if inplace:
#             image.data = shared_variables.last_data
#         else:
#             viewer.add_image(shared_variables.last_data)
#         shared_variables.last_data = None

#     elif shared_variables.last_data is None and shared_variables.raw_data is not None:
#         notifications.show_info('Reverting to raw data')
#         if inplace:
#             image.data = shared_variables.raw_data
#         else:
#             viewer.add_image(shared_variables.raw_data)
#     else:
#         notifications.show_info('Nothing to revert to.')


# # Image corrections
# # use Correct class
# @magic_factory(call_button="Image Corrections")
# def corrections(viewer: Viewer,
#                 image: Image,
#                 hot: Image = None,
#                 std_cutoff: int = 5,
#                 dark: Image = None,
#                 bright: Image = None,
#                 ):
#     original_stack = np.asarray(image.data)
#     # init correction class
#     Correct(hot, std_cutoff, dark, bright)

#     # # first correct bright field for dark
#     # bright_corr = bright.data - dark.data

#     # ans = np.empty(original_stack.shape)
#     # for i, img in enumerate(original_stack):
#     #     ans[i] = img - dark.data - bright_corr

#     # viewer.add_image(ans)

#GT
# def create_layers_class(names):
    
#     class layers_list:
        
#         def __init__(self):
#             self.selected_layers = []
    
#         def add_layer(self, layer):
#             self.selected_layers.append(layer)  
    
#     l_list = layers_list()
    
#     for element in names:
#         l_list.append(element)

# class Oper_modes(Enum):
#     inplace_track = 1         # record history of last operation  (True,True) is from DP, check if needed
#     inplace_NOtrack = 2       # Inplace operations without tracking (dangerous)
    # NOinplace_NOtrack = 3     # Creating copies no tracking needed
    
# from magicgui import magic_factory
# from napari.layers import Layer
# from napari.viewer import current_viewer
# from typing import List

# def current_layers(_):
#     return list(current_viewer().layers)

# from magicgui import magic_factory


# @magic_factory(x={'widget_type': 'Select', "choices": current_layers})
# def example_magic_widget(x: List[Layer]):
#     print(f"you have selected {x}")
    
# if __name__ == '__main__':
#     import napari
    
#     viewer = napari.Viewer()
#     my_widget = example_magic_widget()
#     viewer.window.add_dock_widget(my_widget, name='Widget',
#                                   area='right', add_vertical_stretch=True)
#     napari.run()
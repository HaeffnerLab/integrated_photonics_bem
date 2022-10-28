import numpy as np

#convert rgb value in fusion360 to color index in stl
def _fusion360_RGBA32_to_stlRGB16(r:int,g:int,b:int):
    if r<0 or r>255 or g<0 or g>255 or b<0 or b>255:
        raise ValueError('not a rgb value')
    def _8bit_to_5bit(x):
        if x == 255:
            return 31
        else:
            return int(x/255*31)
    return _8bit_to_5bit(b)*32**2 + _8bit_to_5bit(g)*32 + _8bit_to_5bit(r)

rgb2stl = _fusion360_RGBA32_to_stlRGB16

# standart bemCol rgb values
_bemCol_rgb = {'bem1': (160, 16, 112), 'bem2': (128, 160, 16), 'bem3': (64, 128, 224), 'bem4': (128, 224, 80), 'bem5': (176, 208, 48), 'bem6': (16, 144, 128), 'bem7': (96, 32, 64), 'bem8': (192, 96, 16), 'bem9': (112, 224, 48), 'bem10': (160, 96, 112), 'bem11': (112, 80, 224), 'bem12': (176, 80, 128), 'bem13': (48, 176, 80), 'bem14': (176, 96, 48), 'bem15': (128, 208, 224), 'bem16': (64, 128, 160), 'bem17': (112, 16, 160), 'bem18': (16, 64, 48), 'bem19': (64, 240, 80), 'bem20': (144, 96, 208), 'bem21': (240, 64, 96), 'bem22': (176, 144, 128), 'bem23': (112, 128, 32), 'bem24': (176, 160, 96), 'bem25': (224, 144, 48), 'bem26': (224, 64, 112), 'bem27': (64, 16, 16), 'bem28': (128, 80, 224), 'bem29': (160, 176, 32), 'bem30': (144, 240, 224), 'bem31': (32, 48, 112), 'bem32': (160, 96, 240), 'bem33': (160, 48, 32), 'bem34': (208, 208, 32), 'bem35': (128, 48, 208), 'bem36': (64, 96, 112), 'bem37': (112, 128, 64), 'bem38': (176, 80, 48), 'bem39': (160, 144, 48), 'bem40': (224, 32, 80)}

# a dict {'bemxx':stlColorValue}
bemCol_dict = dict([(_key,rgb2stl(_bemCol_rgb[_key][0],_bemCol_rgb[_key][1],_bemCol_rgb[_key][2])) for _key in _bemCol_rgb.keys()])

'''
or 
bemCol = {'bem1': 13363, 'bem2': 1647, 'bem3': 28135, 'bem4': 10095, 'bem5': 5941, 'bem6': 15905, 'bem7': 7275, 'bem8': 1399, 'bem9': 5997, 'bem10': 13683, 'bem11': 27949, 'bem12': 15669, 'bem13': 9893, 'bem14': 5493, 'bem15': 28463, 'bem16': 19943, 'bem17': 19501, 'bem18': 5345, 'bem19': 10151, 'bem20': 25969, 'bem21': 11517, 'bem22': 15925, 'bem23': 3565, 'bem24': 11893, 'bem25': 5691, 'bem26': 13563, 'bem27': 1063, 'bem28': 27951, 'bem29': 3763, 'bem30': 28593, 'bem31': 13475, 'bem32': 30067, 'bem33': 3251, 'bem34': 3897, 'bem35': 25775, 'bem36': 13671, 'bem37': 7661, 'bem38': 5429, 'bem39': 5683, 'bem40': 9339}
'''

_t = 2**5
meshlab_bemCol = dict(zip(bemCol_dict.keys(),[_t**3 + (_t**2)*(k%_t) + _t*((k//_t)%_t) + k//(_t**2) for k in bemCol_dict.values()]))

'''
{'bem1': 52269, 'bem2': 48737, 'bem3': 40443, 'bem4': 49001, 'bem5': 55077, 'bem6': 34351, 'bem7': 44135, 'bem8': 56673, 'bem9': 46949, 'bem10': 52589, 'bem11': 46395, 'bem12': 54575, 'bem13': 38569, 'bem14': 54629, 'bem15': 48955, 'bem16': 40435, 'bem17': 46131, 'bem18': 34021, 'bem19': 40873, 'bem20': 50553, 'bem21': 62699, 'bem22': 54831, 'bem23': 46563, 'bem24': 54891, 'bem25': 60965, 'bem26': 60653, 'bem27': 39969, 'bem28': 48443, 'bem29': 52899, 'bem30': 51131, 'bem31': 36013, 'bem32': 52605, 'bem33': 52387, 'bem34': 59171, 'bem35': 48313, 'bem36': 40301, 'bem37': 46567, 'bem38': 54565, 'bem39': 52773, 'bem40': 60521}
'''

class my_lib():
    def __init__(self,cl_format):
        self.cl_format = cl_format
        self.lib = {}
        self.description = ''

lib_fusion360 = my_lib(cl_format = ('fusion360','export_stl','RGBA64'))
lib_fusion360.description = '''
    we define a set of standard appearance in fusion 360,
    corresponds to bemCol.adsklib
'''
lib_fusion360.lib = {
    'bem1': (160, 16, 112), 
    'bem2': (128, 160, 16), 
    'bem3': (64, 128, 224), 
    'bem4': (128, 224, 80), 
    'bem5': (176, 208, 48), 
    'bem6': (16, 144, 128), 
    'bem7': (96, 32, 64), 
    'bem8': (192, 96, 16), 
    'bem9': (112, 224, 48), 
    'bem10': (160, 96, 112), 
    'bem11': (112, 80, 224), 
    'bem12': (176, 80, 128), 
    'bem13': (48, 176, 80), 
    'bem14': (176, 96, 48), 
    'bem15': (128, 208, 224), 
    'bem16': (64, 128, 160), 
    'bem17': (112, 16, 160), 
    'bem18': (16, 64, 48), 
    'bem19': (64, 240, 80), 
    'bem20': (144, 96, 208), 
    'bem21': (240, 64, 96), 
    'bem22': (176, 144, 128), 
    'bem23': (112, 128, 32), 
    'bem24': (176, 160, 96), 
    'bem25': (224, 144, 48), 
    'bem26': (224, 64, 112), 
    'bem27': (64, 16, 16), 
    'bem28': (128, 80, 224), 
    'bem29': (160, 176, 32), 
    'bem30': (144, 240, 224), 
    'bem31': (32, 48, 112), 
    'bem32': (160, 96, 240), 
    'bem33': (160, 48, 32), 
    'bem34': (208, 208, 32), 
    'bem35': (128, 48, 208), 
    'bem36': (64, 96, 112), 
    'bem37': (112, 128, 64), 
    'bem38': (176, 80, 48), 
    'bem39': (160, 144, 48), 
    'bem40': (224, 32, 80)
}


class bemColors():
    def __init__(self,_attributes,_format):
        self.electrode_colors = {}
        self.o_stl_attributes, self.o_format = _attributes,_format

        # convert to standard cl_format
        self.stl_attributes, self.cl_format = self.format2RGBA32_list(self.o_stl_attributes, self.o_format)

        # load default libs
        self.col_lib = {}
        self._load_lib(lib_fusion360)
        
    # convert all different cl_format to a standart cl_format in this class: ('RGBA16','_internal')
    def format2RGBA32(self,attr_in:int,format_in):
        attr_out = (None,None,None) ## attr_out = (R,G,B) satisfying 0 <= R,G,B < 32
        format_out = ('RGBA16','_internal')
        if format_in ==  format_out:
            return attr_in,format_in
        elif format_in == ('RGBA16','bgr'):
            order = 2**5
            r = attr_in % order
            g = (attr_in // order) % order
            b = (attr_in // (order ** 2)) % order
            attr_out = (r,g,b)
            return attr_out, format_out
        elif format_in == ('RGBA16','rgb'):
            order = 2**5
            b = attr_in % order
            g = (attr_in // order) % order
            r = (attr_in // (order ** 2)) % order
            attr_out = (r,g,b)
            return attr_out, format_out
        elif format_in == ('meshlab','export_stl'):
            return self.format2RGBA32(attr_in,('RGBA16','rgb'))
        elif format_in == ('fusion360','export_stl'):
            return self.format2RGBA32(attr_in,('RGBA16','bgr'))
        elif format_in == ('fusion360','export_stl','RGBA64'):
            r,g,b = attr_in
            def _8bit_to_5bit(x:int):
                if x < 0 or x> 255:
                    raise ValueError('not a rgb value')
                if x == 255:
                    return 31
                else:
                    return int(x/255*31)
            return (_8bit_to_5bit(r),_8bit_to_5bit(g),_8bit_to_5bit(b)), format_out
        else:
            raise TypeError('format_in not found:'+str(format_in))

    def format2RGBA32_list(self,attr_in,format_in):
        result = [self.format2RGBA32(attr_in[i],format_in) for i in range(len(attr_in))]
        at, fo = zip(*result)
        return list(at), fo[0]


    def _load_lib(self,new_lib):
        cl_format = new_lib.cl_format
        for name,value in new_lib.lib.items():
            self.set_my_color(value,cl_format,name)

    def set_my_color(self,value,cl_format,name,set_uknown = False):
        if name in self.col_lib:
            raise ValueError('name existed')
        elif name[0] == '_' and not set_uknown:
            raise ValueError('Dont set a name with _ the first character yourself')
        else:
            self.col_lib[name] = self.format2RGBA32(value,cl_format)[0]

    def print_stl_colors(self):
        '''
        Print the colors in the stl model.
        '''
        # to collect unknown colors
        unknown_colors = []
        print('COLORS in the stl:')
        for attr in self.stl_attributes:
            print_name = []
            for lib_name,lib_attr in self.col_lib.items():
                if attr == lib_attr:
                    print_name.append(lib_name)
            if len(print_name) == 0:
                unknown_colors.append(attr)
            else:
                print(print_name)
        for i in range(len(unknown_colors)):
            print_name = ['_unkColor'+str(i)]
            self.set_my_color(unknown_colors[i],('RGBA16','_internal'),print_name[0],set_uknown=True)
            print(print_name)
        print('TOTAL COLORS: ', len(self.stl_attributes))

    def color_electrode(self,color ,name ):
        '''
        color: the color of an electrode surface (e.g. bem1)
        name: the name of the electrode (e.g. DC1)
        '''
        if self.col_lib[color] in self.electrode_colors:
            raise ValueError('rgb key collision')
        # search origin color encoding corresponding to self.col_lib[color]
        if self.col_lib[color] not in list(self.stl_attributes):
            raise ValueError('invalid color')
        o_attr = self.o_stl_attributes[list(self.stl_attributes).index(self.col_lib[color])]
        self.electrode_colors[o_attr] = name

    def drop_colors(self):
        '''
        Drops colors that are not assigned a name
        '''
        num = 0
        for o_attr in self.o_stl_attributes:
            if o_attr not in self.electrode_colors:
                print_name = []
                for lib_name,lib_attr in self.col_lib.items():
                    if self.format2RGBA32(o_attr,self.o_format)[0] == lib_attr:
                        print_name.append(lib_name)
                if len(print_name) != 0:
                    print('dropping color',print_name)
                    num += 1
        print('TOTAL COLORS DROPPED: ', num)

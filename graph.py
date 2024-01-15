import matplotlib.pyplot as plt


def table1():
    vgg_loss_bwn = {
                "FWN": [0.960487, 0.788119, 0.659242, 0.578226, 0.625066, 0.553263, 0.645237, 0.645596, 0.632455, 0.686648, 0.71462, 0.658474, 0.636229, 0.707946, 0.705443, 0.831792, 0.684302, 0.754243, 0.70312, 0.76003, 0.775344, 0.711623, 0.785931, 0.710541, 0.749973, 0.763617, 0.745495, 0.867526, 0.75918, 0.676197, 0.57125, 0.571012, 0.578207, 0.576727, 0.580702, 0.579194, 0.588094, 0.585718, 0.586521, 0.586123, 0.585125, 0.581724, 0.58063, 0.580269, 0.576041, 0.577842, 0.576244, 0.570819, 0.574398, 0.571916, 0.565528, 0.568931, 0.56439, 0.563767, 0.557225, 0.560808, 0.563384, 0.55456, 0.554379, 0.552973, 0.556238, 0.554582, 0.553168, 0.556132, 0.553119, 0.553905, 0.55236, 0.556944, 0.555759, 0.554256, 0.552939, 0.552585, 0.556483, 0.552213, 0.551172, 0.550492, 0.553831, 0.553977, 0.556837, 0.549052, 0.550693, 0.554111, 0.554927, 0.555127, 0.548569, 0.550105, 0.550516, 0.550396, 0.549061, 0.553926],
                "BWN": [1.212222, 0.854992, 0.753442, 0.726969, 0.633607, 0.686317, 0.729406, 0.866169, 0.707975, 0.772517, 0.718552, 0.822317, 1.011975, 0.83925, 0.685529, 0.718167, 0.761612, 0.753451, 0.696317, 0.785281, 0.758666, 0.803148, 0.911875, 0.733659, 0.770685, 0.706081, 0.723084, 0.746057, 0.845389, 0.758307, 0.586664, 0.593618, 0.599286, 0.610719, 0.618316, 0.625328, 0.638117, 0.640042, 0.650499, 0.655952, 0.661151, 0.665724, 0.668542, 0.677631, 0.673784, 0.679539, 0.681905, 0.685536, 0.685911, 0.692869, 0.695231, 0.701069, 0.702842, 0.704825, 0.711418, 0.707161, 0.70978, 0.716371, 0.711679, 0.707357, 0.712628, 0.713849, 0.713686, 0.712948, 0.715453, 0.716987, 0.718888, 0.711727, 0.71634, 0.713501, 0.717343, 0.712073, 0.720634, 0.713812, 0.715973, 0.720379, 0.719513, 0.716156, 0.718914, 0.717254, 0.722227, 0.724272, 0.725006, 0.720269, 0.720727, 0.721112, 0.720099, 0.720541, 0.719817, 0.724519],
                "SQ_BWN": [2.043666, 1.299895, 1.701256, 2.034739, 0.841577, 1.036974, 1.378255, 2.109034, 1.700519, 1.094453, 1.029271, 1.79964, 1.776693, 0.958756, 0.715657, 1.717571, 1.622371, 3.088793, 0.951804, 0.844085, 0.865476, 0.824712, 1.199487, 1.069921, 1.092972, 1.647076, 1.475679, 2.573205, 1.63222, 1.219053, 0.573021, 0.755212, 0.610095, 0.615798, 0.727228, 0.644543, 0.667103, 0.676368, 0.685547, 0.695217, 0.708071, 0.718473, 0.720256, 0.727151, 0.723807, 0.782191, 0.75326, 0.69064, 0.709115, 0.714876, 0.718568, 0.728611, 0.736258, 0.734485, 0.744166, 0.746352, 0.753858, 0.758453, 0.755098, 0.756449, 0.755742, 0.758426, 0.757229, 0.761113, 0.761238, 0.75591, 0.758133, 0.713519, 0.711809, 0.711726, 0.711977, 0.711864, 0.715688, 0.712074, 0.713744, 0.716335, 0.7117, 0.714211, 0.71303, 0.714868, 0.713986, 0.714456, 0.720021, 0.718943, 0.717639, 0.715025, 0.720591, 0.717096, 0.717382, 0.720487],
               }
    vgg_loss_twn = {
                "FWN": [0.960487, 0.788119, 0.659242, 0.578226, 0.625066, 0.553263, 0.645237, 0.645596, 0.632455, 0.686648, 0.71462, 0.658474, 0.636229, 0.707946, 0.705443, 0.831792, 0.684302, 0.754243, 0.70312, 0.76003, 0.775344, 0.711623, 0.785931, 0.710541, 0.749973, 0.763617, 0.745495, 0.867526, 0.75918, 0.676197, 0.57125, 0.571012, 0.578207, 0.576727, 0.580702, 0.579194, 0.588094, 0.585718, 0.586521, 0.586123, 0.585125, 0.581724, 0.58063, 0.580269, 0.576041, 0.577842, 0.576244, 0.570819, 0.574398, 0.571916, 0.565528, 0.568931, 0.56439, 0.563767, 0.557225, 0.560808, 0.563384, 0.55456, 0.554379, 0.552973, 0.556238, 0.554582, 0.553168, 0.556132, 0.553119, 0.553905, 0.55236, 0.556944, 0.555759, 0.554256, 0.552939, 0.552585, 0.556483, 0.552213, 0.551172, 0.550492, 0.553831, 0.553977, 0.556837, 0.549052, 0.550693, 0.554111, 0.554927, 0.555127, 0.548569, 0.550105, 0.550516, 0.550396, 0.549061, 0.553926],
                "TWN": [1.047445, 0.767734, 0.767838, 0.715518, 0.672362, 0.648284, 0.582462, 0.621748, 0.625219, 0.778775, 0.653818, 0.747543, 0.676199, 0.74138, 0.706625, 0.76542, 0.781185, 0.76351, 0.704309, 0.724926, 0.861624, 0.71804, 0.793625, 0.729211, 0.909537, 0.728613, 0.74747, 0.707012, 0.652756, 0.707474, 0.576882, 0.5812, 0.590548, 0.597449, 0.608737, 0.619448, 0.624705, 0.632546, 0.639899, 0.644936, 0.653915, 0.660592, 0.6659, 0.670188, 0.675376, 0.680639, 0.679865, 0.684726, 0.694475, 0.692787, 0.697086, 0.700446, 0.705285, 0.704786, 0.704781, 0.708278, 0.70561, 0.713998, 0.713819, 0.717835, 0.715362, 0.721708, 0.717714, 0.71949, 0.71612, 0.719639, 0.720538, 0.720324, 0.71838, 0.719793, 0.720639, 0.719577, 0.717956, 0.719248, 0.721686, 0.725155, 0.718828, 0.719389, 0.720043, 0.727595, 0.725328, 0.723832, 0.724496, 0.725701, 0.724616, 0.727004, 0.728731, 0.725757, 0.73128, 0.720645],
                "SQ_TWN": [1.057837, 0.848616, 0.867521, 0.915524, 0.790551, 0.695129, 0.69276, 0.735676, 0.797435, 0.641774, 0.772973, 0.697223, 0.699963, 0.846779, 0.765016, 0.661085, 0.818224, 0.826487, 0.910741, 1.137332, 1.109674, 1.037114, 0.945025, 1.108085, 0.845072, 0.870062, 0.842372, 1.053678, 0.846256, 0.950255, 0.588816, 0.598115, 0.588022, 0.602495, 0.601686, 0.594795, 0.600481, 0.602221, 0.612457, 0.629508, 0.613473, 0.61412, 0.612815, 0.621457, 0.61917, 0.624677, 0.619986, 0.616829, 0.614981, 0.619204, 0.616033, 0.609636, 0.61811, 0.613027, 0.615816, 0.610094, 0.612458, 0.605407, 0.610008, 0.606971, 0.603446, 0.605639, 0.607718, 0.604908, 0.604049, 0.606167, 0.602767, 0.602179, 0.599314, 0.600627, 0.601432, 0.599564, 0.601668, 0.601665, 0.598873, 0.595041, 0.594378, 0.601901, 0.60253, 0.601189, 0.600671, 0.599377, 0.598242, 0.598523, 0.599835, 0.597513, 0.60472, 0.599386, 0.602551, 0.597373]
               }
    vgg_loss_ttq = {
                "FWN": [0.960487, 0.788119, 0.659242, 0.578226, 0.625066, 0.553263, 0.645237, 0.645596, 0.632455, 0.686648, 0.71462, 0.658474, 0.636229, 0.707946, 0.705443, 0.831792, 0.684302, 0.754243, 0.70312, 0.76003, 0.775344, 0.711623, 0.785931, 0.710541, 0.749973, 0.763617, 0.745495, 0.867526, 0.75918, 0.676197, 0.57125, 0.571012, 0.578207, 0.576727, 0.580702, 0.579194, 0.588094, 0.585718, 0.586521, 0.586123, 0.585125, 0.581724, 0.58063, 0.580269, 0.576041, 0.577842, 0.576244, 0.570819, 0.574398, 0.571916, 0.565528, 0.568931, 0.56439, 0.563767, 0.557225, 0.560808, 0.563384, 0.55456, 0.554379, 0.552973, 0.556238, 0.554582, 0.553168, 0.556132, 0.553119, 0.553905, 0.55236, 0.556944, 0.555759, 0.554256, 0.552939, 0.552585, 0.556483, 0.552213, 0.551172, 0.550492, 0.553831, 0.553977, 0.556837, 0.549052, 0.550693, 0.554111, 0.554927, 0.555127, 0.548569, 0.550105, 0.550516, 0.550396, 0.549061, 0.553926],
                "TTQ":[2.358558, 2.364615, 2.31706, 2.310261, 2.311288, 2.304947, 2.305006, 2.303931, 2.161198, 1.909849, 1.777865, 1.59076, 1.616246, 1.350692, 1.25326, 1.278003, 1.124768, 1.02743, 1.020229, 0.966322, 1.000262, 0.984136, 0.94831, 0.986575, 0.96183, 0.900195, 0.932773, 0.93838, 0.919387, 0.976178, 0.905993, 1.022591, 1.127303, 1.225281, 1.293831, 1.366993, 1.402386, 1.44544, 1.477501, 1.510917, 1.537633, 1.569072, 1.58725, 1.625458, 1.638075, 1.651088, 1.679214, 1.657741, 1.673335, 1.690678, 1.684778, 1.687083, 1.68564, 1.70872, 1.717411, 1.735759, 1.715684, 1.738221, 1.754857, 1.72886, 1.720905, 1.725711, 1.736903, 1.744774, 1.717493, 1.729709, 1.733661, 1.721956, 1.722309, 1.715722, 1.710459, 1.719142, 1.73053, 1.71626, 1.740994, 1.715523, 1.707735, 1.737365, 1.73814, 1.722384, 1.724391, 1.72556, 1.744022, 1.718382, 1.729425, 1.733129, 1.733811, 1.74345, 1.71463, 1.717509], 
                "SQ_TTQ": [2.124618, 1.852496, 1.777299, 1.437106, 1.158009, 1.054357, 0.905531, 0.84694, 0.883679, 0.802048, 0.767621, 0.822594, 0.786585, 0.861704, 0.836484, 1.01293, 0.986133, 0.928628, 0.953729, 0.977718, 0.962142, 1.196208, 0.974623, 1.123039, 0.992004, 1.080868, 1.010611, 1.012795, 0.946446, 1.014346, 0.848658, 0.880317, 0.923993, 0.947269, 0.965944, 0.980332, 0.9908, 0.997927, 1.016403, 1.011887, 1.018022, 1.021422, 1.035444, 1.033645, 1.035926, 1.034827, 1.041736, 1.02561, 1.036567, 1.036345, 1.038068, 1.044328, 1.037568, 1.029707, 1.034943, 1.033583, 1.035462, 1.024705, 1.029442, 1.03138, 1.03245, 1.023895, 1.028652, 1.032156, 1.029674, 1.027556, 1.038523, 1.03889, 1.02567, 1.030097, 1.039274, 1.024077, 1.025986, 1.031002, 1.040349, 1.02874, 1.019464, 1.021651, 1.021845, 1.036304, 1.024836, 1.016828, 1.024119, 1.025883, 1.023212, 1.025479, 1.019118, 1.03137, 1.023716, 1.028099]
               }
    resnet_loss_bwn = {
                   "FWN": [1.241012, 1.031331, 0.921509, 0.686017, 0.789444, 0.633988, 0.715641, 0.626931, 0.676506, 0.876581, 0.756744, 0.701622, 0.746686, 0.670924, 0.827641, 0.703811, 0.632216, 0.699558, 0.822384, 1.048731, 0.631711, 1.021135, 0.702879, 0.827358, 0.760324, 0.833864, 0.772303, 0.984107, 0.779302, 0.90424, 0.579629, 0.586874, 0.600442, 0.60635, 0.614564, 0.624958, 0.629196, 0.634193, 0.639679, 0.646331, 0.64782, 0.650521, 0.654113, 0.654817, 0.658891, 0.662221, 0.663155, 0.663428, 0.659131, 0.662478, 0.666394, 0.665433, 0.662089, 0.668518, 0.670466, 0.66526, 0.67135, 0.665466, 0.668654, 0.665023, 0.668535, 0.67156, 0.670389, 0.665286, 0.664392, 0.665202, 0.665748, 0.665949, 0.670042, 0.66991, 0.671943, 0.668608, 0.67057, 0.666393, 0.669378, 0.666709, 0.666597, 0.67074, 0.669112, 0.664939, 0.66533, 0.665842, 0.666487, 0.670988, 0.665787, 0.664439, 0.662041, 0.666848, 0.667047, 0.665873],
                   "BWN": [1.386978, 1.650947, 1.091556, 0.990543, 0.98179, 0.866026, 1.760496, 0.800284, 0.950989, 0.773307, 0.868049, 0.755898, 0.787684, 0.742987, 0.733594, 0.766711, 0.798516, 0.835757, 0.715708, 0.775956, 0.81769, 0.851681, 0.827658, 0.679672, 0.88715, 0.71567, 0.713494, 0.794103, 0.827718, 0.70318, 0.534001, 0.536521, 0.53721, 0.541078, 0.542424, 0.546771, 0.549017, 0.551586, 0.551583, 0.55046, 0.554738, 0.554501, 0.55609, 0.559391, 0.557726, 0.556515, 0.561921, 0.560089, 0.561412, 0.563276, 0.561429, 0.562176, 0.563821, 0.564821, 0.565974, 0.567684, 0.564855, 0.561927, 0.562771, 0.566791, 0.562079, 0.561144, 0.561963, 0.561707, 0.564351, 0.566924, 0.566172, 0.560707, 0.563292, 0.563736, 0.562367, 0.562945, 0.56244, 0.562052, 0.564404, 0.565198, 0.564275, 0.561878, 0.56452, 0.566459, 0.565332, 0.564976, 0.564981, 0.564246, 0.566802, 0.565395, 0.564548, 0.566111, 0.564892, 0.565434],
                   "SQ_BWN": [2.194787, 2.366196, 2.245379, 1.845977, 3.322699, 1.706103, 1.571155, 1.911368, 1.579081, 1.309399, 2.118453, 1.244411, 1.459398, 1.217124, 1.248158, 2.39915, 1.293264, 1.584167, 7.053225, 2.015883, 1.498605, 1.542315, 2.396544, 1.018604, 1.091996, 1.259987, 1.506989, 1.180121, 1.088552, 0.957356, 0.595592, 0.519857, 0.552613, 0.522183, 0.527898, 0.565323, 0.538414, 0.548759, 0.594072, 0.5663, 0.562603, 0.570611, 0.57082, 0.600592, 0.610634, 0.625743, 0.661559, 0.605046, 0.585086, 0.578243, 0.58312, 0.571402, 0.639252, 0.626828, 0.590216, 0.583229, 0.577342, 0.571215, 0.590797, 0.576973, 0.570195, 0.551943, 0.600139, 0.563538, 0.561509, 0.564882, 0.563647, 0.563677, 0.563546, 0.562739, 0.56068, 0.560389, 0.562635, 0.561396, 0.560142, 0.561263, 0.561225, 0.562964, 0.560766, 0.560701, 0.558921, 0.561216, 0.561368, 0.560665, 0.560591, 0.558443, 0.558282, 0.56239, 0.557421, 0.559754],
                }
    resnet_loss_twn = {
                   "FWN": [1.241012, 1.031331, 0.921509, 0.686017, 0.789444, 0.633988, 0.715641, 0.626931, 0.676506, 0.876581, 0.756744, 0.701622, 0.746686, 0.670924, 0.827641, 0.703811, 0.632216, 0.699558, 0.822384, 1.048731, 0.631711, 1.021135, 0.702879, 0.827358, 0.760324, 0.833864, 0.772303, 0.984107, 0.779302, 0.90424, 0.579629, 0.586874, 0.600442, 0.60635, 0.614564, 0.624958, 0.629196, 0.634193, 0.639679, 0.646331, 0.64782, 0.650521, 0.654113, 0.654817, 0.658891, 0.662221, 0.663155, 0.663428, 0.659131, 0.662478, 0.666394, 0.665433, 0.662089, 0.668518, 0.670466, 0.66526, 0.67135, 0.665466, 0.668654, 0.665023, 0.668535, 0.67156, 0.670389, 0.665286, 0.664392, 0.665202, 0.665748, 0.665949, 0.670042, 0.66991, 0.671943, 0.668608, 0.67057, 0.666393, 0.669378, 0.666709, 0.666597, 0.67074, 0.669112, 0.664939, 0.66533, 0.665842, 0.666487, 0.670988, 0.665787, 0.664439, 0.662041, 0.666848, 0.667047, 0.665873],
                   "TWN": [1.59909, 1.322641, 0.869375, 0.859697, 0.807684, 0.941199, 0.857663, 0.825389, 0.708925, 0.647893, 0.984417, 0.783969, 0.723621, 0.776829, 0.680427, 0.901151, 0.6272, 0.78466, 0.721694, 0.715832, 0.809919, 0.790694, 0.945259, 0.749839, 0.652688, 0.670057, 0.719102, 0.680353, 0.652923, 0.842959, 0.528031, 0.52741, 0.531337, 0.534939, 0.543969, 0.543489, 0.547492, 0.55386, 0.552787, 0.556286, 0.559171, 0.563298, 0.564088, 0.564791, 0.565409, 0.564976, 0.572542, 0.568492, 0.572835, 0.574093, 0.57252, 0.571353, 0.575375, 0.574218, 0.577548, 0.578207, 0.574637, 0.581927, 0.578788, 0.579851, 0.578153, 0.581884, 0.576494, 0.575809, 0.579558, 0.579921, 0.577479, 0.578301, 0.580693, 0.578596, 0.579714, 0.57744, 0.579531, 0.580448, 0.580117, 0.582589, 0.581945, 0.581108, 0.584117, 0.581968, 0.582816, 0.580325, 0.584286, 0.579899, 0.582926, 0.580999, 0.579999, 0.581108, 0.585164, 0.582228],
                   "SQ_TWN": [1.559935, 1.358294, 1.131145, 0.949488, 0.835794, 1.22372, 1.501096, 0.874906, 0.972363, 1.183415, 1.297601, 0.788382, 0.841636, 0.764634, 1.211267, 0.696774, 1.042678, 1.527529, 1.515438, 1.598865, 0.736598, 0.840894, 0.90301, 0.871061, 1.036483, 1.14594, 0.951102, 0.930145, 0.879845, 1.175486, 0.570525, 0.536878, 0.553612, 0.600023, 0.570949, 0.597577, 0.607958, 0.70471, 0.704122, 0.692019, 0.633263, 0.67684, 0.690358, 0.656819, 0.660258, 0.663295, 0.761283, 0.685158, 0.694895, 0.753891, 0.666504, 0.677815, 0.731675, 0.711586, 0.696536, 0.717273, 0.694888, 0.691557, 0.741597, 0.696752, 0.702253, 0.686885, 0.691565, 0.691822, 0.688434, 0.686817, 0.699219, 0.695347, 0.691816, 0.693594, 0.688986, 0.693785, 0.690236, 0.691129, 0.696555, 0.696034, 0.6944, 0.693332, 0.698302, 0.690712, 0.691973, 0.695856, 0.695435, 0.692498, 0.696507, 0.697399, 0.691601, 0.694206, 0.694483, 0.697458]
                }
    resnet_loss_ttq = {
                   "FWN": [1.241012, 1.031331, 0.921509, 0.686017, 0.789444, 0.633988, 0.715641, 0.626931, 0.676506, 0.876581, 0.756744, 0.701622, 0.746686, 0.670924, 0.827641, 0.703811, 0.632216, 0.699558, 0.822384, 1.048731, 0.631711, 1.021135, 0.702879, 0.827358, 0.760324, 0.833864, 0.772303, 0.984107, 0.779302, 0.90424, 0.579629, 0.586874, 0.600442, 0.60635, 0.614564, 0.624958, 0.629196, 0.634193, 0.639679, 0.646331, 0.64782, 0.650521, 0.654113, 0.654817, 0.658891, 0.662221, 0.663155, 0.663428, 0.659131, 0.662478, 0.666394, 0.665433, 0.662089, 0.668518, 0.670466, 0.66526, 0.67135, 0.665466, 0.668654, 0.665023, 0.668535, 0.67156, 0.670389, 0.665286, 0.664392, 0.665202, 0.665748, 0.665949, 0.670042, 0.66991, 0.671943, 0.668608, 0.67057, 0.666393, 0.669378, 0.666709, 0.666597, 0.67074, 0.669112, 0.664939, 0.66533, 0.665842, 0.666487, 0.670988, 0.665787, 0.664439, 0.662041, 0.666848, 0.667047, 0.665873],
                   "TTQ": [1.661096, 1.356166, 1.387207, 1.132596, 1.217358, 0.989637, 0.990417, 1.136018, 0.883527, 0.875487, 0.912797, 0.993777, 0.789812, 0.937085, 0.8496, 0.838708, 0.746985, 0.915675, 0.905824, 0.76302, 0.833391, 0.832264, 0.836599, 0.861649, 0.796252, 1.005463, 0.887806, 0.840725, 0.98656, 1.069499, 0.570358, 0.595951, 0.623236, 0.641136, 0.659594, 0.687243, 0.697823, 0.723684, 0.739972, 0.75338, 0.774269, 0.799615, 0.798439, 0.812191, 0.822678, 0.842091, 0.841627, 0.853882, 0.859278, 0.858329, 0.872007, 0.890894, 0.886701, 0.893808, 0.898412, 0.901165, 0.91072, 0.908609, 0.917004, 0.917248, 0.91198, 0.915421, 0.912421, 0.913023, 0.912295, 0.91445, 0.918059, 0.921057, 0.91921, 0.913183, 0.91754, 0.915627, 0.915709, 0.916734, 0.919571, 0.916731, 0.916251, 0.91739, 0.924948, 0.918925, 0.92368, 0.919953, 0.922916, 0.922567, 0.922063, 0.919531, 0.928425, 0.925755, 0.923713, 0.922283],
                   "SQ_TTQ": [1.298633, 1.006204, 0.826706, 0.801738, 0.727554, 0.776982, 0.682691, 0.850192, 0.77579, 0.680572, 0.77112, 0.737697, 0.823884, 0.705272, 0.844362, 0.774999, 0.900774, 0.752278, 0.80267, 0.953974, 1.107683, 0.735693, 0.840567, 0.813635, 0.792616, 0.827481, 0.869433, 0.913376, 0.948888, 0.874774, 0.608773, 0.624767, 0.640437, 0.655758, 0.667215, 0.681807, 0.695022, 0.697767, 0.706486, 0.709423, 0.715768, 0.720652, 0.726822, 0.72866, 0.733169, 0.735788, 0.744818, 0.744865, 0.74269, 0.745614, 0.748473, 0.74483, 0.752293, 0.750622, 0.750746, 0.755767, 0.754173, 0.757931, 0.756116, 0.754514, 0.758553, 0.758986, 0.758963, 0.756037, 0.759809, 0.758822, 0.755738, 0.756082, 0.760229, 0.755277, 0.753413, 0.755045, 0.759814, 0.760668, 0.759424, 0.759815, 0.757214, 0.757282, 0.758879, 0.757273, 0.760417, 0.75452, 0.753913, 0.753445, 0.757866, 0.762133, 0.759468, 0.759016, 0.760483, 0.758252]
                }
   
    return vgg_loss_bwn, vgg_loss_twn, vgg_loss_ttq, resnet_loss_bwn, resnet_loss_twn, resnet_loss_ttq


if __name__ == "__main__":
    d1, d2, d3, d4, d5, d6 = table1()
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16, 8), constrained_layout=True)
    fig.tight_layout()

    for k, v in d1.items():
        xs = range(0, len(v))
        ys = v
        ax[0, 0].plot(xs, ys, label=f"{k}")
        ax[0, 0].set_title(f"VGG-9 model with quant BWN")

    for k, v in d2.items():
        xs = range(0, len(v))
        ys = v
        ax[0, 1].plot(xs, ys, label=f"{k}")
        ax[0, 1].set_title(f"VGG-9 model with quant TWN")

    for k, v in d3.items():
        xs = range(0, len(v))
        ys = v
        ax[0, 2].plot(xs, ys, label=f"{k}")
        ax[0, 2].set_title(f"VGG-9 model with quant TTQ")

    for k, v in d4.items():
        xs = range(0, len(v))
        ys = v
        ax[1, 0].plot(xs, ys, label=f"{k}")
        ax[1, 0].set_title(f"ResNet-20 model with quant BWN")

    for k, v in d5.items():
        xs = range(0, len(v))
        ys = v
        ax[1, 1].plot(xs, ys, label=f"{k}")
        ax[1, 1].set_title(f"ResNet-20 model with quant TWN")

    for k, v in d6.items():
        xs = range(0, len(v))
        ys = v
        ax[1, 2].plot(xs, ys, label=f"{k}")
        ax[1, 2].set_title(f"ResNet-20 model with quant TTQ")

    for axes in ax.flat:
        axes.set(xlabel = "epoch", ylabel="loss")
        axes.legend()
        axes.grid()
        axes.set_xlim([0, 90])

    plt.legend()
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(f"graphs.png", dpi=300)
    plt.show()

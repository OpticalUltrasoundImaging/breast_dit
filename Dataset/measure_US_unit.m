clear all
[path,~]=imgetfile();

img = imread(path);
figure;imshow(img)
h_hor = images.roi.Line(gca,'Position',[100 100;0 100]);
pixel_per_unit_hor = abs(h_hor.Position(1,1)-h_hor.Position(2,1))

figure;imshow(img)
h_ver = images.roi.Line(gca,'Position',[100 100;100 0]);
pixel_per_unit_hor = abs(h_ver.Position(1,2)-h_ver.Position(2,2))
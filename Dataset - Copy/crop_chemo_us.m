clear all

folder_path = uigetdir();
dirs = dir(folder_path);
dfolders = dirs([dirs(:).isdir]) ;
dfolders = dfolders(~ismember({dfolders(:).name},{'.','..'}));
rect = [573,424,398,303];
img_type = '*.png';
for i = 1:length(dfolders)
    current_folder = dfolders(i).folder;
    final_path = dfolders(i).name;
    detail_path = split(current_folder,'\');
    img_dir = dir(fullfile(current_folder,final_path,img_type));
    img_name = {img_dir(:).name};
    us_dirs = dir(fullfile(current_folder,final_path));
    us_dir = us_dirs([us_dirs(:).isdir]);
    us_dir = us_dir(~ismember({us_dir(:).name},{'.','..'}));
    us_name = {us_dir(:).name};
    US_name = us_name(contains(us_name,'US'));

    patien_id = 'NIRUS';
    index_patient = detail_path{contains(detail_path,patien_id)};
    folder = 'C:\Users\Minghao\Box\DOT\chemo_US';
    us_folder = [];
    cyc_folder = [];
    if ~isempty(US_name) % if have us_only folder, read us_only first, then target US
        us_folder = [];
        cyc_folder = final_path;
        new_folder = fullfile(folder, index_patient,cyc_folder,us_folder);
        if ~exist(new_folder, 'dir')
            mkdir(new_folder)
        end
        txt_name = {img_dir(:).name};
        tar_name = txt_name(contains(txt_name,'Target'));
        if ~isempty(tar_name)
            load_len = length(tar_name);
        else
            load_len = length(img_dir);
        end
        for id = 1:load_len
            if startsWith( img_dir(id).name , '.')
                continue
            end
            if ~isempty(tar_name)
                I = imread(fullfile(img_dir(id).folder,tar_name{id}));
            else
                I = imread(fullfile(img_dir(id).folder,img_dir(id).name));
            end
            if isempty(rect)
                [J,rect] = imcrop(I);
                rect = round(rect);
                if ~isempty(tar_name)
                    [~,split_img,ext] = fileparts(tar_name{id});
                else
                    [~,split_img,ext] = fileparts(img_dir(id).name);
                end
                imwrite(J,fullfile(new_folder,[split_img,'_x',num2str(rect(1)),',y',num2str(rect(2)),',w',num2str(rect(3)),',h',num2str(rect(4)),ext]))
            else
                figure(2);imshow(I);
                h = imrect(gca, rect);pos = wait(h);
                hold on; rectangle('position', pos, 'EdgeColor', 'g', 'LineWidth', 2)
                img2 = imcrop(I, round(pos));
                pos = round(pos);
                figure(3); imshow(img2, []);
                if ~isempty(tar_name)
                    [~,split_img,ext] = fileparts(tar_name{id});
                else
                    [~,split_img,ext] = fileparts(img_dir(id).name);
                end
                imwrite(img2,fullfile(new_folder,[split_img,'_x',num2str(pos(1)),',y',num2str(pos(2)),',w',num2str(pos(3)),',h',num2str(pos(4)),ext]))
                close all
            end
        end
        
        
        us_folder = 'US_only';
        
        new_folder = fullfile(folder, index_patient,cyc_folder,us_folder);
        if ~exist(new_folder, 'dir')
            mkdir(new_folder)
        end
        img_dir = dir(fullfile(current_folder,final_path,US_name{1},img_type));
        for id = 1:length(img_dir)
            if startsWith( img_dir(id).name , '.')
                continue
            end
            I = imread(fullfile(img_dir(id).folder,img_dir(id).name));
            if isempty(rect)
                [J,rect] = imcrop(I);
                rect = round(rect);
                [~,split_img,ext] = fileparts(img_dir(id).name);
                imwrite(J,fullfile(new_folder,[split_img,'_x',num2str(rect(1)),',y',num2str(rect(2)),',w',num2str(rect(3)),',h',num2str(rect(4)),ext]))
            else
                figure(2);imshow(I);
                h = imrect(gca, rect);pos = wait(h);
                hold on; rectangle('position', pos, 'EdgeColor', 'g', 'LineWidth', 2)
                img2 = imcrop(I, round(pos));
                pos = round(pos);
                figure(3); imshow(img2, []);
                [~,split_img,ext] = fileparts(img_dir(id).name);
                imwrite(img2,fullfile(new_folder,[split_img,'_x',num2str(pos(1)),',y',num2str(pos(2)),',w',num2str(pos(3)),',h',num2str(pos(4)),ext]))
                close all
            end
        end
        
        
    else
        cyc_folder = final_path;
        
        new_folder = fullfile(folder, index_patient,cyc_folder,us_folder);
        if ~exist(new_folder, 'dir')
            mkdir(new_folder)
        end
        txt_name = {img_dir(:).name};
        tar_name = txt_name(contains(txt_name,'Target'));
        if ~isempty(tar_name)
            load_len = length(tar_name);
        else
            load_len = length(img_dir);
        end
        
        for id = 1:load_len
            if startsWith( img_dir(id).name , '.')
                continue
            end
            if ~isempty(tar_name)
                I = imread(fullfile(img_dir(id).folder,tar_name{id}));
            else
                I = imread(fullfile(img_dir(id).folder,img_dir(id).name));
            end
            if isempty(rect)
                [J,rect] = imcrop(I);
                rect = round(rect);
                if ~isempty(tar_name)
                    [~,split_img,ext] = fileparts(tar_name{id});
                else
                    [~,split_img,ext] = fileparts(img_dir(id).name);
                end
                imwrite(J,fullfile(new_folder,[split_img,'_x',num2str(rect(1)),',y',num2str(rect(2)),',w',num2str(rect(3)),',h',num2str(rect(4)),ext]))
            else
                figure(2);imshow(I);
                h = imrect(gca, rect);pos = wait(h);
                hold on; rectangle('position', pos, 'EdgeColor', 'g', 'LineWidth', 2)
                img2 = imcrop(I, round(pos));
                pos = round(pos);
                figure(3); imshow(img2, []);
                if ~isempty(tar_name)
                    [~,split_img,ext] = fileparts(tar_name{id});
                else
                    [~,split_img,ext] = fileparts(img_dir(id).name);
                end
                imwrite(img2,fullfile(new_folder,[split_img,'_x',num2str(pos(1)),',y',num2str(pos(2)),',w',num2str(pos(3)),',h',num2str(pos(4)),ext]))
                close all
            end
        end
    end
    

    
end
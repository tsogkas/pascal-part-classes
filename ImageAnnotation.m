classdef ImageAnnotation < handle
% IMAGEANNOTATION Information about the objects in an image (inherits from handle).
%   IMAGEANNOTATION includes information about the objects contained in
%   an image, their parts and the image itself. It also offers several
%   easy-to-use functions to augment or modify image annotations (such as
%   create a resized or rotated version of the image and the objects
%   included.
%
% Properties
%   dataset             % dataset from which the image was taken
%   imname              % name of the image file
%   imsize              % size of the image
%   imdir               % image directory
%   objects             % array of PascalObject objects
%   isflipped = false;  % horizontal flip flag
%   theta = 0;          % rotation angle in degrees (~=0 if rotated)
%   scale = 1;          % image scale (~=1 if resized)
% 
% anno = ImageAnnotation() creates an empty ImageAnnotation object.
% anno = ImageAnnotation(arg) creates a new Image annotation object and 
%   copies valid field values from arg which can be either a struct
%   containing annotation information or another ImageAnnotation object.
% 
% NOTE: ImageAnnotation inherits from class handle. This makes it very 
%   efficient since all operations are performed using references and not
%   object copies. However this also means that special care should be
%   taken, especially when copying objects or passing an object as a
%   function argument. Please see the examples below for more information. 
%
% Examples:
%   s.imname = 'testImage'; s.scale = 2;
%   obj1 = ImageAnnotation(s);    % create ImageAnnotation object from struct
%   obj2 = ImageAnnotation(obj1); % create new obj2 and copy field values from obj1
%                               
%   obj3 = obj1;     % only copies the reference, it DOES NOT create a new object
%   obj3 == obj1     % this returns true (1). obj3 and obj1 are the same object.
%   obj3.scale = 1;  % now obj1.scale has changed too!
%   obj3 = ImageAnnotation(obj1); 
%   obj3.scale = 1;            % now obj3 is a new object with the same field
%                              % values as obj1, except its scale.
% 
%   foo(obj1);                 % if obj1 is modified in foo the changes persist!
%   foo(ImageAnnotation(obj1)) % this way we apply foo in a new object that is
%                              % a copy of obj1 and leave obj1 unchanged.
% 
% See also: PascalObject, PascalPart
% 
% Stavros Tsogkas, <stavros.tsogkas@centralesupelec.fr>
% Last update: March 2015

    properties
        dataset             % dataset from which the image was taken
        imname              % name of the image file
        imsize              % size of the image
        imdir               % image directory
        objects             % array of PascalObject objects
        isflipped = false;  % image is flipped
        theta = 0;          % image is rotated by angle theta (in degrees)
        scale = 1;          % image is resized by a factor or to a defined size
    end
    
    methods
        function anno = ImageAnnotation(src, imdir)  % TODO: update help 
            if nargin > 0
                if isa(src,'ImageAnnotation')
                    anno = copyFromImageAnnotationObject(anno,src);
                elseif isstruct(src)
                    anno = copyFromStruct(anno, src);
                elseif ischar(src)
                    if isdir(src)
                        anno = copyFromDirectory(anno,src);
                    else
                        anno = copyFromFile(anno, src);
                    end
                elseif iscell(src) && all(cellfun(@ischar, src))
                    anno = copyFromFileList(anno, src);
                else
                    error('Input must be a struct, a file name, or an ImageAnnotation object')
                end
                if nargin == 2
                    assert(ischar(imdir), 'Image directory path must be a string')
                    [anno(:).imdir] = deal(imdir);
                end
            end
        end
        
        % Manipulate image annotations and containing objects
        function anno = flip(anno)
            % FLIP  Horizontally flip ImageAnnotation.
            %   Object and part masks are flipped horizontally and part 
            %   names are changed to their symmetric, based on the
            %   PascalObject.part2symmetric() map. E.g. 'larm' (left arm)
            %   becomes 'rarm' (right arm), 'leye' (left eye) becomes 'reye'
            %   etc.
            %
            %   anno = FLIP(anno) 
            %
            %   See also: ImageAnnotation.rotate, ImageAnnotation.resize, 
            %       PascalObject.mask2bbox, PascalObject.part2symmetric, fliplr.
            symmetricPart = PascalObject.part2symmetric();
            for i=1:length(anno.objects)
                if ~isempty(anno.objects(i).mask)
                    anno.objects(i).mask = fliplr(anno.objects(i).mask);
                    if ~isempty(anno.objects(i).bbox)
                        anno.objects(i).bbox = mask2bbox(anno.objects(i));
                    end
                end
                for j=1:length(anno.objects(i).parts)
                    if ~isempty(anno.objects(i).parts(j).mask)
                        anno.objects(i).parts(j).mask = ...
                            fliplr(anno.objects(i).parts(j).mask);
                        if ~isempty(anno.objects(i).parts(j).bbox)
                            anno.objects(i).parts(j).bbox = ...
                                mask2bbox(anno.objects(i).parts(j));
                        end
                    end
                    if symmetricPart.isKey(anno.objects(i).parts(j).class)
                        anno.objects(i).parts(j).class = ...
                            symmetricPart(anno.objects(i).parts(j).class);
                    end
                end
            end
            anno.isflipped = ~anno.isflipped;
        end   
        function anno = rotate(anno,theta)  
            % ROTATE Rotate image annotation.
            %   Rotates masks for objects and their parts and
            %   (re)calculates the rotated object and part centroids.
            %   WARNING!: This can be very slow if the number of objects is
            %   large.
            %
            %   anno = ROTATE(anno,theta)  rotate anno theta degrees.  
            %
            %   See also: imrotate, PascalObject.rotate, ImageAnnotation.resize,
            %       ImageAnnotation.flip  
            rotate(anno.objects,theta)
            anno.theta  = theta;
            anno.imsize = size(anno.objects(1).mask);
        end
        function anno = resize(anno,scale)
            % RESIZE  Resize image annotation.
            %   Resizes masks for objects and their parts and
            %   (re)calculates the resized object and part centroids.
            %
            %   anno = RESIZE(anno,scale)  scale is a scalar or a
            %       two-element vector (same as for imresize).
            %
            %   See also: PascalObject.resize, imresize,
            %     ImageAnnotation.rotate, ImageAnnotation.flip.
            resize(anno.objects, scale);
            anno.scale  = scale;
            anno.imsize = size(anno.objects(1).mask);
        end
        function [im, cobjs] = crop(anno, varargin)
            % CROP  Crop image boxes of objects or object parts.
            %   
            %   im = CROP(anno) crops all object boxes from the image
            %
            %   im = CROP(anno, 'PropertyName',PropertyValue) gives more
            %   control over the objects and the parts cropped.
            %
            %   [im,cobjs] = CROP(anno) also returns handles (pointers) to
            %   the objects/parts cropped.
            %
            %   Properties
            %   'objects':  string or cell array of strings with valid
            %               object classes, or a vector of integers. If the 
            %               value of this property is empty ([]), then all 
            %               the objects are cropped (default: []).
            %   'parts':    string or cell array of strings with valid part
            %               classes, or 'all'. If the value of this
            %               property is empty ([]), then no parts are
            %               cropped and the function returns only cropped
            %               objects (default: []).
            %   'pad':      a scalar corresponding to the padding around 
            %               the cropped image rectangle. If pad is smaller
            %               than 1, then the padding is proportional to
            %               min(imageHeight,imageWidth).
            %
            %   Examples:
            %   im = CROP(anno,'objects',2);       % crops the 2nd object in anno
            %   im = CROP(anno,'objects',[1,3,5]); % crops the 1st, 3rd and 5th object in anno
            %   im = CROP(anno,'objects','car');   % crops only objects that are cars
            %   im = CROP(anno,'objects',{'car','dogs'}); % crops objects that are cars or dogs
            %   im = CROP(anno,'objects','person','parts','leye'} % crops left eyes from 'person' objects
            %   im = CROP(anno,'parts','leye'}     % crops left eyes for all classes           
            %   subclasses = PascalObject.class2subclasses();
            %   im = CROP(anno,'parts',subclasses('eye')} % crops all eyes for all classes 
            %
            %   See also: PascalObject.class2subclasses, padarray,
            %       ImageAnnotation.readImage
%             warning('This function has not been thoroughly tested!')
            validArgs = {'pad','objects','parts'};
            defaultValues = {0,[],[]};
            vmap = parseVarargin(varargin,validArgs,defaultValues);
            objects_ = vmap('objects');
            parts_   = vmap('parts');
            pad      = vmap('pad');
            if isempty(objects_)        % crop all objects
                cobjs = anno.objects;
            elseif iscell(objects_)     % crop objects of chosen class(es)
                cobjs = anno.objects(ismember({anno.objects(:).class},objects_));
            elseif ischar(objects_)     % crop objects of chosen class
                cobjs = anno.objects(strcmp(objects_, {anno.objects(:).class}));
            elseif isvector(objects_)   % crop objects with specified indexes
                cobjs = anno.objects(objects_);
            else
                error(['Value of property ''objects'' must be a string, '... 
                    'a cell array of strings, or a vector of integers'])
            end
            if ~isempty(parts_)
                cobjs = [cobjs(:).parts];
                if strcmp(parts_,'all') % crop all parts (do nothing)
                elseif iscell(parts_)   % crop parts of chosen class(es)
                    cobjs = cobjs(ismember({cobjs(:).class}, parts_));
                elseif ischar(parts_)   % crop parts of chosen class
                    cobjs = cobjs(strcmp(parts_, {cobjs(:).class}));
                else
                    error(['Value of property ''parts'' must be ''all'', '...
                        'a part class string, or a cell array of strings.'])
                end
            end
            pads   = zeros(4,1);
            imFull = readImage(anno);
            im     = cell(1,numel(cobjs));
            for i=1:numel(cobjs);
                bb = getbbox(cobjs(i));
                if pad > 0 && pad < 1   % padding as dimension percentage
                    height = bb(4)-bb(2)+1;
                    width  = bb(3)-bb(1)+1;
                    pad    = round(pad*min(height,width));
                end
                pads(1) = bb(1) - pad - 1;  
                pads(2) = bb(2) - pad - 1;
                pads(3) = anno.imsize(2) - bb(3) - pad;
                pads(4) = anno.imsize(1) - bb(4) - pad;
                pads(pads>0) = 0;   % if pads > 0, then we do not need to pad
                pads  = abs(pads);  % (padding falls inside the image)
                bb(1) = max(bb(1)-pad, 1);  % adjust bounding box
                bb(2) = max(bb(2)-pad, 1);
                bb(3) = min(bb(3)+pad, anno.imsize(2));
                bb(4) = min(bb(4)+pad, anno.imsize(1));
                im{i} = imFull(bb(2):bb(4), bb(1):bb(3), :);
                if any(pads)
                    im{i} = padarray(im{i}, pads([2,1]), 'replicate','pre');
                    im{i} = padarray(im{i}, pads([4,3]), 'replicate','post');
                end
            end
            cobjs = crop(cobjs,[],pad);
        end
        
        % Plotting functions
        function im = readImage(anno)
            % im = READIMAGE(anno)  Read image from a struct or
            %   ImageAnnotation object. Checks if the annotation is
            %   resized/rotated/flipped and modifies the original image
            %   accordingly. This function first checks if anno.imname is
            %   the full path to the image and reads it. If not, it uses
            %   the paths stored in anno.imdir (if any) and anno.imname to
            %   construct the full path and read the image.
            %
            % See also: imread, imrotate, flipdim.
            [~,~,ext] = fileparts(anno.imname);
            if isempty(ext)
                imName = [anno.imname, '.jpg'];
            else
                imName = anno.imname;
            end
            try     % First try to see if anno.name is the complete path
                im = imread(imName);
            catch
                % Otherwise use the directory path imdir
                try
                    im = imread(fullfile(anno.imdir, imName));
                catch
                    error('File does not exist.')
                end
            end
            if anno.theta,      im = imrotate(im,anno.theta,'bilinear'); end
            if anno.scale ~= 1, im = imresize(im,anno.scale,'bilinear'); end
            if anno.isflipped,  im = flipdim(im,2); end
        end 
        function showImage(anno)
            % SHOWIMAGE Display ImageAnnotation image.
            %   
            %   SHOWIMAGE(anno)
            %
            %   See also: ImageAnnotation.readImage, imshow
            
            % TODO: add pre-processing functionality
            imshow(readImage(anno));
        end  
        function drawBoxes(anno, obj, varargin) %TODO: add private function to handle color input (e.g. 'r', 'red', [1 0 0] should all be valid)
            % DRAWBOXES  Draw bounding boxes for objects and parts on image
            %
            %   DRAWBOXES(anno)  draws bounding boxes for all objects in anno.
            %
            %   DRAWBOXES(anno,obj)  draws the bounding box only for the
            %       PascalObject obj.
            %
            %   DRAWBOXES(anno,obj,'PropertyName',PropertyValue) 
            %
            %   Properties:
            %   'color':     'r','b','g' etc. (as in plot)
            %   'lineWidth':  width for the box lines (default: 4).
            %   'showImage':  show the image and then draw. Set to false if
            %                 you want to keep drawing on the same figure
            %                 (default: true).
            %   'showTitle'   prints a title. By default this is the image 
            %                 id. The user can enter a custom title.
            %                 (default: true)
            %   'showParts':  show bounding boxes for parts too (default: false).
            %
            %   See also: drawBoxes, parseVarargin, ImageAnnotation.drawMasks
            %   PascalObject.class2color, ImageAnnotation.showImage
            if nargin < 2 || isempty(obj), obj = anno.objects; end;
            validArgs = {'color','lineWidth','showImage','showTitle','showParts'};
            defaultValues = {[],4,true,false,false};
            vmap = parseVarargin(varargin, validArgs,defaultValues);
            if vmap('showImage'), showImage(anno); end % by default refresh image
            if vmap('showTitle')
                if ischar(vmap('showTitle'))
                    title(vmap('showTitle'));
                else
                    title(['Image ' anno.imname],'interpreter','none');
                end
            end
            hold on;
            bboxColor = PascalObject.class2color();
            for i=1:numel(obj)
                if vmap('color')
                    color = vmap('color');
                else
                    color = bboxColor(obj(i).class);
                end
                drawBoxes(getbbox(obj(i)),'color',color,'lineWidth',vmap('lineWidth'));
                % draw part boxes with half the line width
                if vmap('showParts') && ~isempty(obj(i).parts) 
                    partBoxes = cat(1, obj(i).parts(:).bbox);
                    drawBoxes(partBoxes,'color',color,'lineWidth',vmap('lineWidth')/2);
                end
            end
            hold off;
        end
        function drawMasks(anno, obj, varargin)
            % DRAWMASKS Overlay colored masks for objects and their parts 
            %   on the original image.
            % 
            %   DRAWMASKS(anno)  draws masks for all objects in anno.
            %
            %   DRAWMASKS(anno,obj)  draws masks only for the
            %       PascalObject obj.
            %
            %   DRAWMASKS(anno,obj,'PropertyName',PropertyValue) 
            %
            %   Properties:
            %   'color':      'r','b','g' etc. (as in plot)
            %   'markerType': marker type used (default: '.').
            %   'markerSize': marker size used (default: 5).
            %   'showImage':  show the image and then draw. Set to false if
            %                 you want to keep drawing on the same figure
            %                 (default: true).
            %   'showTitle'   prints a title. By default this is the image 
            %                 id. The user can enter a custom title 
            %                 (default: true).            
            %   'showParts':  show masks for parts too (default: false).
            %   'showBoundaries': show boundaries for objects and parts 
            %                   (default: false).
            %
            %   See also: drawMasks, parseVarargin, ImageAnnotation.drawBoxes
            %   PascalObject.class2color, ImageAnnotation.showImage
            
            if nargin < 2 || isempty(obj), obj = anno.objects; end;
            validArgs = {'color','markerType','markerSize','showImage',...
                'showTitle','showParts','showBoundaries'};
            defaultValues = {[],'.',5,true,false,false,false};
            vmap = parseVarargin(varargin, validArgs,defaultValues);
            if vmap('showImage'), showImage(anno); end    % by default refresh image
            if vmap('showTitle'), title(['Image ' anno.imname],'interpreter','none'); end
            hold on;
            maskColor = PascalObject.class2color(); 
            cmap      = PascalObject.colormap();
            for i=1:numel(obj)
                % Plot object mask (for some classes, the object is not
                % fully composed of its parts so it is better to draw the
                % object mask too. 
                if vmap('color')
                    color = vmap('color');
                else
                    color = maskColor(obj(i).class);
                end
                [y,x] = find(obj(i).mask);
                plot(x,y,vmap('markerType'),'Color',color,'MarkerSize',vmap('markerSize'));
                if vmap('showBoundaries') % draw object boundaries
                    [y,x] = find(bwperim(obj(i).mask)); plot(x,y,'.k'); 
                end
                
                % Plot part masks
                if vmap('showParts') && ~isempty(obj(i).parts)
                    mask = mergeMasks(obj(i).parts);    
                    vals = unique(mask);
                    %cmap = jet(max(vals)); % colormap for parts (old)
                    for v=2:numel(vals)     % 1st entry is background (0)
                        [y,x] = find(mask==vals(v));
                        partColorIndex = PascalObject.nClasses + 1 + vals(v);
                        plot(x,y,vmap('markerType'),'Color',cmap(partColorIndex,:),...
                            'MarkerSize',vmap('markerSize'));
                    end
                    if vmap('showBoundaries')  % draw part boundaries
                        [y,x] = find(edge(mask,'sobel',0.0001)); plot(x,y,'.k'); 
                    end
                end
            end
            hold off;
        end
        function drawFullAnnotation(anno)
            % DRAWFULLANNOTATION  Draw everything: boxes and masks for all
            %   objects in the image and their parts, using default settings.
            %
            %   DRAWFULLANNOTATION(anno)
            %
            %   See also: ImageAnnotation.drawBoxes, ImageAnnotation.drawMasks
            %drawBoxes(anno,anno.objects, varargin{:});
            drawMasks(anno, anno.objects, 'showParts',true);
            drawBoxes(anno, anno.objects, 'showImage',false);
        end
        
    end
    
    methods(Access = private)
        % Copy constructors to read ImageAnnotation data from input 
        function anno = copyFromImageAnnotationObject(anno,a)
            nImages = numel(a);
            anno(nImages) = ImageAnnotation();  % Preallocate
            % Copy all property values except objects
            props = properties(ImageAnnotation); props(strcmp(props,'objects')) = [];
            for i=1:numel(props)
                [anno(:).(props{i})] = a(:).(props{i});
            end
            for i=1:nImages % copy objects
                if ~isempty(a(i).objects)
                    anno(i).objects = PascalObject(a(i).objects); 
                end
            end
        end
        function anno = copyFromStruct(anno,s)
            oidFields = {'meta'; 'image'; 'aeroplane';...
                'verticalStabilizer'; 'nose'; 'wing'; 'wheel'};
            pascalPartsFields = {'imname'; 'objects'};
            if isequal(pascalPartsFields,fieldnames(s))
                anno = copyFromPascalPartsStruct(anno,s);
            elseif isequal(oidFields,fieldnames(s))
                anno = copyFromOIDStruct(anno,s);
            else
                error('Struct does not comply with the OID or PascalParts format')
            end
        end
        function anno = copyFromPascalPartsStruct(anno,s)
            anno.dataset = 'PascalParts';
            anno.imname  = s.imname;                % image name
            anno.objects = PascalObject(s.objects); % objects
            if ~isempty(anno.objects)
                anno.imsize = size(anno.objects(1).mask); % image size
                passInfoToObjects(anno);
            end
        end
        function anno = copyFromOIDStruct(anno,oid)
            nImages = numel(oid.image.id);
            anno(nImages) = ImageAnnotation();
            [anno(:).dataset] = deal('OID');
            [anno(:).imname] = oid.image.name{:};
            c = mat2cell(oid.image.size', ones(size(oid.image.size,2),1), 2);
            [anno(:).imsize] = c{:};
            aeroplanes = OIDAeroplane(oid);
            aeroplaneParentId = oid.aeroplane.parentId;
            imageId = oid.image.id;
            for i=1:nImages;
                anno(i).objects = aeroplanes(aeroplaneParentId == imageId(i));
                passInfoToObjects(anno(i));
            end
        end
        function anno = copyFromFile(anno,file)
            tmp  = load(file);
            if isstruct(tmp.anno)
                anno = copyFromStruct(anno, tmp.anno);
            elseif isa(tmp.anno, 'ImageAnnotation')
                anno = copyFromImageAnnotationObject(anno, tmp.anno);
            else
                error('File must contain a struct or an ImageAnnotation object called ''anno''')
            end
        end
        function anno = copyFromFileList(anno, list)
            % list is a cell array containing file names
            nFiles = numel(list);
            disp(['Copying from ' num2str(nFiles) ' annotation files...']);
            anno(nFiles) = ImageAnnotation();
            for i=1:nFiles
                anno(i) = copyFromFile(anno(i), list{i});
            end
        end
        function anno = copyFromDirectory(anno,d)
            % d is a path to a directory
            disp(['Loading annotations from ' d ' directory']);
            matFiles = dir(fullfile(d, '*.mat'));
            if d(end) ~= filesep, d(end+1) = filesep; end
            anno = copyFromFileList(anno, strcat(d,{matFiles(:).name}));
        end
        function passInfoToObjects(anno)
            objects_ = anno.objects; 
            parts_   = cat(2, objects_(:).parts);
            if ~isempty(anno.imname)
                [objects_(:).imname] = deal(anno.imname);
                [parts_(:).imname]   = deal(anno.imname);
            end
            if ~isempty(anno.imsize)
                [objects_(:).imsize] = deal(anno.imsize);
                [parts_(:).imsize]   = deal(anno.imsize);                
            end
        end
        
    end
    methods(Hidden) % Mostly legacy methods
        function [im, cobjs]  = cropObjects(anno, obj, pad)
            % LEGACY FUNCTION, USE ImageAnnotation.crop instead
            % Returns a cell array of tightly cropped images around the objects
            % obj: Object can have the following forms:
            %      1) a vector of indices of the objects to crop
            %      2) a string of an object or part class name
            %      3) a cell array of strings with the object or part names
            %      4) a pair of cell arrays {{},{}}, where the first cell
            %      array contains the part names and the second the object
            %      names
            % pad: padding around cropped bounding box. Can take either a
            %      fixed value, or the string value 'adjust', in which
            %      case, the padding around the cropped box is 0.2*
            %      min(height,width) of the part/object bounding box
            
            warning('This is a legacy function. Use ImageAnnotation.crop instead')
            % Parse input arguments
            if nargin < 3, pad = 0; end;
            if nargin < 2 || isempty(obj)  % crop all objects
                cobjs = anno.objects;
            elseif iscell(obj)
                if numel(obj) == 2 && iscell(obj{1}) && iscell(obj{2})  
                    % crop parts belonging to specified object classes
                    objNames = obj{2};
                    partNames = obj{1};
                    objs  = anno.objects(ismember({anno.objects(:).class},objNames));
                    parts = [objs(:).parts];
                    if isempty(parts)
                        cobjs = [];
                    else
                        cobjs = parts(ismember({parts(:).class}, partNames));
                    end
                else    % list of part/object names
                    isObject = ismember(obj, PascalObject.validObjectClasses());
                    assert(all(isObject) || ~any(isObject), ...
                        ['Mixing object and part queries is not allowed.\n',...
                        'If you want to retrieve parts of a specific object class',...
                        ' use a cell array of cells {{partName},{objName}} as input.'])
                    objs = anno.objects(:);
                    if all(isObject)
                        cobjs = objs(ismember({objs(:).class}, obj));
                    elseif isempty([objs(:).parts])
                        cobjs = [];
                    else
                        parts = [objs(:).parts];
                        cobjs = parts(ismember({parts(:).class}, obj));
                    end
                end
            elseif ischar(obj) || (iscell(obj) && numel(obj) == 1)
                if iscell(obj), obj = obj{1}; end
                isObject = any(strcmp(obj, PascalObject.validObjectClasses));
                if isObject         % crop objects of specified class
                    cobjs = anno.objects(strcmp(obj, {anno.objects(:).class}));
                else                % crop parts of specified class
                    parts = [anno.objects(:).parts];
                    if isempty(parts)
                        cobjs = [];
                    else
                        cobjs = parts(strcmp(obj, {parts(:).class}));
                    end
                end
            elseif isnumeric(obj)    % crop objects with specified indexes
                cobjs = anno.objects(obj);
            else
                error('obj must be a string, a cell array of strings, or an index vector')
            end
            pads = zeros(4,1);
            imFull = readImage(anno);
            im = cell(numel(cobjs), 1);
            for i=1:numel(cobjs);
                bb = getbbox(cobjs(i));
                if strcmp(pad,'adjust') % set pad to 20% of the smallest dimension
                    height = bb(4)-bb(2)+1;
                    width  = bb(3)-bb(1)+1;
                    pad = round(0.2*min(height,width));
                end
                pads(1) = bb(1) - pad - 1;
                pads(2) = bb(2) - pad - 1;
                pads(3) = anno.imsize(2) - bb(3) - pad;
                pads(4) = anno.imsize(1) - bb(4) - pad;
                pads(pads>0) = 0;
                pads  = abs(pads);
                bb(1) = max(bb(1)-pad, 1);
                bb(2) = max(bb(2)-pad, 1);
                bb(3) = min(bb(3)+pad, anno.imsize(2));
                bb(4) = min(bb(4)+pad, anno.imsize(1));
                im{i} = imFull(bb(2):bb(4), bb(1):bb(3), :);
                im{i} = padarray(im{i}, pads([2,1]), 'replicate','pre');
                im{i} = padarray(im{i}, pads([4,3]), 'replicate','post');
            end
        end  
        function [im, cparts] = cropParts(anno, obj, pad)
            % LEGACY FUNCTION, USE ImageAnnotation.crop instead
            % Returns a cell array of tightly cropped images around the
            % object parts
            % obj: can have the following forms:
            %      1) a vector of indices of the objects whose parts to crop
            %      2) a string of an object or part class name
            %      3) a cell array of strings with the object or part names
            %      4) a pair of cell arrays {{},{}}, where the first cell
            %      array contains the part names and the second the object
            %      names
            % pad: padding around cropped bounding box. Can take either a
            %      fixed value, or the string value 'adjust', in which
            %      case, the padding around the cropped box is 0.2*
            %      min(height,width) of the part/object bounding box
            
            warning('This is a legacy function. Use ImageAnnotation.crop instead')
            % Parse input arguments
            if nargin < 3, pad = 0; end;
            if nargin < 2 || isempty(obj)  % crop all objects
                cparts = [anno.objects(:).parts];
            elseif iscell(obj)
                if numel(obj) == 2 && iscell(obj{1}) && iscell(obj{2})  
                    % crop parts belonging to specified object classes
                    objNames = obj{2};
                    partNames = obj{1};
                    objs  = anno.objects(ismember({anno.objects(:).class},objNames));
                    cparts = [objs(:).parts];
                    if ~isempty(cparts)
                        cparts = cparts(ismember({cparts(:).class}, partNames));
                    end
                else    % list of part/object names
                    cparts = [anno.objects(:).parts];
                    cparts = cparts(ismember({cparts(:).class}, obj));
                end
            elseif ischar(obj) || (iscell(obj) && numel(obj) == 1)
                if iscell(obj), obj = obj{1}; end
                cparts = [anno.objects(:).parts];
                if ~isempty(cparts)
                    cparts = cparts(strcmp(obj, {cparts(:).class}));
                end
            elseif isnumeric(obj)    % crop objects with specified indexes
                cparts = [anno.objects(obj).parts];
            else
                error('obj must be a string, a cell array of strings, or an index vector')
            end
            pads = zeros(4,1);
            imFull = readImage(anno);
            im = cell(numel(cparts), 1);
            for i=1:numel(cparts);
                bb = getbbox(cparts(i));
                if strcmp(pad,'adjust') % set pad to 20% of the smallest dimension
                    height = bb(4)-bb(2)+1;
                    width  = bb(3)-bb(1)+1;
                    pad = round(0.2*min(height,width));
                end
                pads(1) = bb(1) - pad - 1;
                pads(2) = bb(2) - pad - 1;
                pads(3) = anno.imsize(2) - bb(3) - pad;
                pads(4) = anno.imsize(1) - bb(4) - pad;
                pads(pads>0) = 0;
                pads  = abs(pads);
                bb(1) = max(bb(1)-pad, 1);
                bb(2) = max(bb(2)-pad, 1);
                bb(3) = min(bb(3)+pad, anno.imsize(2));
                bb(4) = min(bb(4)+pad, anno.imsize(1));
                im{i} = imFull(bb(2):bb(4), bb(1):bb(3), :);
                im{i} = padarray(im{i}, pads([2,1]), 'replicate','pre');
                im{i} = padarray(im{i}, pads([4,3]), 'replicate','post');
            end
        end  
    end
end

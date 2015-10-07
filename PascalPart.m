classdef PascalPart < PascalObject
% PascalPart  An object part (inherits from classes PascalObject, handle)
%   This class inherits from PascalObject so documentation is very similar
%   to the PascalObject documentation. This class, as well as PASCALOBJECT
%   and IMAGEANNOTATION classes were inspired and based on the Pascal-Parts
%   dataset from [1].
%
%   [1]: Detect What You Can: Detecting and Representing Objects using
%        Holistic Models and Body Parts, CVPR 2014
%
%   Properties:
%       objectClass:class of the object containing the part (string)
%       class:      class name  (string)
%       classIndex: class index (integer/double)
%       mask:       object mask (logical)
%       parts:      object parts (array of PascalPart objects)
%       bbox:       object bounding box (1x4 vector of integers/doubles)
%       centroid    object centroid coordinates (double)
%       imname      name of the image containing the object (string)
%   
%
%   obj = PASCALPART()  creates an empty PascalPart object
%   obj = PASCALPART(arg)  creates an new PascalPart object
%       and copies part information from arg. Input arg can be a
%       part class string, a struct or another PascalPart
%       object. If arg is a valid object class string, a minimal
%       PascalPart is created. If arg is a struct or a
%       PascalPart object, information from valid fields such as
%       the object bounding box, the object mask, centroids etc.,
%       are copied into the new object; otherwise, the constructor
%       tries to compute them from what information is available.
%   
%   NOTE: PASCALPART inherits from class handle. This makes it very 
%       efficient since all operations are performed using references and not
%       object copies. However this also means that special care should be
%       taken, especially when copying objects or passing an object as a
%       function argument. Please see the examples below for more information. 
%
%   Examples:
%       obj = PASCALPART();        % create empty object
%       obj = PASCALPART('leye');  % create empty 'left eye' object
%
%       s.class = 'leye'; s.mask = someBinaryMask;
%       obj1 = PASCALPART(s);     % create left eye part from struct
%       obj2 = PASCALPART(obj1);  % create new obj2 and copy field values from obj1
%                               
%       obj3 = obj1; % only copies the reference, it DOES NOT create a new object
%       obj3 == obj1 % this returns true (1). obj3 and obj1 are the same object.
%       obj3.class = 'head';  % now obj1.class has changed too!
%       obj3 = PASCALPART(obj1); 
%       obj3.class = 'head';    % now obj3 is a new object with the same field
%                               % values as obj1, except its class.
% 
%       foo(obj1);              % if obj1 is modified in foo the changes persist!
%       foo(PASCALPART(obj1)) % this way we apply foo in a new object that is
%                               % a copy of obj1 and obj1 remains unchanged
%
%   See also: PascalObject, PascalObject.part2index, PascalObject.mask2bbox
%
%   Stavros Tsogkas, <stavros.tsogkas@ecp.fr>
%   Last update: November 2014

    properties
        objectClass
    end
    
    methods
        function part = PascalPart(arg)
            if nargin > 0                 
                if isstruct(arg) 
                    part = copyFromPascalPartsStruct(part, arg);
                elseif isa(arg,'PascalPart')
                    part = copyFromPascalPart(part, arg);
                elseif ischar(arg)
                    part.class = arg;
                else
                    error('Input argument must be a struct, a PascalPart or a string')
                end
            end
        end
        
        function m = mergeMasks(parts, bbox, pad, multiLabel)
            % MERGEMASKS  Merge part binary masks into a single mask.
            %   By default, MERGEMASKS combines part masks to obtain a
            %   multi-label union mask. Since parts can overlap or entirely
            %   cover one-another, it uses the occlusion ordering taken by
            %   PascalObject.part2occlusion().
            %
            %   m = MERGEMASKS(parts)
            %
            %   m = MERGEMASKS(parts,bbox) merges part masks and crops the
            %       part of the result that corresponds to the bounding box 
            %       bbox. bbox can also be empty ([]), in which case, no
            %       cropping takes place (default: []).
            %
            %   m = MERGEMASKS(parts,bbox,pad) also zero-pads the merged mask 
            %       with pad pixels (default: 0).
            %       
            %   m = MERGEMASKS(parts,bbox,pad,multiLabel) sets multi-label
            %       option. If multiLabel is 0/false then the merged mask is
            %       just the union of the parts' binary masks (default: true). 
            %   
            %   See also: PascalObject.part2occlusion, 
            %       PascalObject.part2index, padarray
            if nargin < 2, bbox = []; end
            if nargin < 3, pad = 0; end
            if nargin < 4, multiLabel = true; end  %TODO: consider replacing multiLabel flag with a mergeParts map
            if multiLabel   % get merged labels
                % We use a depth map based on the occlusion ordering
                % defined in PascalObject.part2occlusion(), to determine
                % which label to use in case of parts that overlap.
                depth    = PascalObject.part2occlusion(parts(1).objectClass);
                m        = zeros(size(parts(1).mask),'uint8');
                depthMap = zeros(size(parts(1).mask),'uint8');
                for i=1:numel(parts)
                    p = parts(i);
                    m(p.mask & (depthMap <= depth(p.class))) = p.classIndex;
                    depthMap(m == p.classIndex) = depth(p.class); 
                end
            else
                m = sum(cat(3, parts(:).mask), 3) > 0;
            end
            if ~isempty(bbox) % crop around the OBJECT bbox
                m = m(bbox(2):bbox(4), bbox(1):bbox(3));
            end
            if pad  % pad with zeros
                m = padarray(m, [pad pad], 0);
            end
        end 
        function computeCentroids(part)
            % COMPUTECENTROIDS Computes and stores part centroids.
            %
            %   COMPUTECENTROIDS(part) where part is a PascalPart object.
            %
            % See also: regionprops, PascalObject.getbbox
            for j=1:numel(part)     % for the object itself
                assert(~isempty(part(j).mask), 'Part mask is empty')
                props = regionprops(part(j).mask, 'centroid');
                part(j).centroid = props.Centroid;
            end
        end

    end
    
    methods(Access = private)
        function parts = copyFromPascalPartsStruct(parts, s)
            nParts = numel(s);
            parts(nParts)    = PascalPart();
            [parts(:).class] = s(:).part_name;
            [parts(:).mask]  = s(:).mask;
            for i=1:nParts % TODO: maybe make this optional to avoid fragmentation
                parts(i).bbox = mask2bbox(parts(i).mask);
            end
        end
        % TODO: CONSIDER REPLACING WITH copyFromPascalObject
        function parts = copyFromPascalPart(parts,pp)
            parts(numel(pp)) = PascalPart();
            props = properties(PascalPart);
            for i=1:numel(props)
                [parts(:).(props{i})] = pp(:).(props{i});
            end
        end
    end
    
    methods(Hidden)
        function part = PascalPartLegacy(arg)
            if nargin > 0     
                if isstruct(arg) || isa(arg,'PascalPart')
                    nParts = length(arg);
                    part(nParts) = PascalPart(); % Preallocate object array
                    for i=1:nParts
                        if isa(arg, 'PascalPart')           % PascalPart
                            part(i).class = arg(i).class;
                        elseif isfield(arg(i),'part_name')  % struct based on [1]
                            part(i).class = arg(i).part_name;
                        end
                        part(i).mask = arg(i).mask;         % part mask
                        if isfield(arg(i),'bbox') && ~isempty(arg(i).bbox)
                            part(i).bbox = arg(i).bbox;
                        elseif ~isempty(part(i).mask)
                            part(i).bbox = mask2bbox(part(i));
                        end
                    end
                elseif ischar(arg)
                    part.class = arg;
                else
                    error('Input argument must be a struct, a PascalPart or a string')
                end
            end
        end
    end
end

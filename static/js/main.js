$(document).ready(function () {
    // Init
    $('.image-section').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }

    }
    function readURL1(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview1').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview1').hide();
                $('#imagePreview1').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }

    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        readURL(this);
    });
    $("#imageUpload1").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        readURL1(this);
    });

    

});

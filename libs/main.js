function onReady() {
    document.getElementById('burger-toggle').addEventListener('click', () => {
        const elem = document.getElementById('burger-menu')
        if (getComputedStyle(elem, null).visibility == 'hidden') {
            elem.style.visibility = 'visible'
        } else {
            elem.style.visibility = 'hidden'
        }
    })

    const emailIcon = document.getElementById('email-icon')

    if(emailIcon !== null) {
        document.getElementById('email-icon').addEventListener('click', () => {
            // encoded to avoid email scrapers
            const email = atob('a2VubnlAZmFsa2Flci5pbw==')
            const elem = document.getElementById('email-box')
            if (getComputedStyle(elem, null).visibility == 'hidden') {
                const ref = document.getElementById('email-ref')
                ref.innerHTML = email
                ref.href = "mailto:" + email
                elem.style.visibility = 'visible'
            } else {
                elem.style.visibility = 'hidden'
            }
        })
    }
}

document.addEventListener('DOMContentLoaded', onReady, false);

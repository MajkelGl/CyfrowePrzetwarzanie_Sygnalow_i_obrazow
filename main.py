from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure
from scipy.signal import convolve2d
from scipy.ndimage import median_filter, minimum_filter, maximum_filter, uniform_filter, gaussian_filter, convolve
from skimage.filters import rank
from skimage.morphology import footprint_rectangle
from skimage.util import img_as_ubyte

def zadanie_5():
    # Wczytywanie obrazu
    nazwa_pliku = input("Podaj nazwę pliku do wczytania (np. aerial_view.tif): ")
    try:
        im = Image.open(nazwa_pliku)
        d = np.array(im)
    except FileNotFoundError:
        print("Błąd: Plik nie został znaleziony.")
        return

    # Wybór profilu jasności
    kierunek = input("Wybierz kierunek profilu (h - poziomy, v - pionowy): ").lower()
    wsp = int(input("Podaj współrzędną (X lub Y) linii: "))

    if kierunek == 'h' and 0 <= wsp < d.shape[0]:
        greyness = d[wsp, :]
        x = np.linspace(0, d.shape[1] - 1, num=d.shape[1])
    elif kierunek == 'v' and 0 <= wsp < d.shape[1]:
        greyness = d[:, wsp]
        x = np.linspace(0, d.shape[0] - 1, num=d.shape[0])
    else:
        print("Błąd: Niepoprawna opcja lub współrzędna poza zakresem!")
        return

    # Wykres jasności
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.imshow(d, cmap='gray')
    plt.title("Wczytany obraz")
    plt.axis('off')

    plt.subplot(2, 1, 2)
    plt.plot(x, greyness, color='black')
    plt.xlabel("Pozycja")
    plt.ylabel("Poziom jasności")
    if kierunek == 'h':
        plt.title("Profil jasności obrazu " + nazwa_pliku + ", wedlug poziomej linii " + str(wsp))
    elif kierunek == 'v':
        plt.title("Profil jasności obrazu " + nazwa_pliku + ", wedlug pionowej linii " + str(wsp))
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    czy_wyswietlic = input("Czy zapisac wycinek? 1 -> TAK, 0-> NIE")
    if czy_wyswietlic == '1':
        # Wyświetlenie zakresów
        print(f"\nRozmiar obrazu: szerokość = {d.shape[1]}, wysokość = {d.shape[0]}")
        print("Zakres współrzędnych X: 0 do", d.shape[1] - 1)
        print("Zakres współrzędnych Y: 0 do", d.shape[0] - 1)

        # Wybór i zapis podobrazu
        wsp_x1 = int(input("Podaj X1 do wycięcia: "))
        wsp_y1 = int(input("Podaj Y1 do wycięcia: "))
        wsp_x2 = int(input("Podaj X2 do wycięcia: "))
        wsp_y2 = int(input("Podaj Y2 do wycięcia: "))
        nazwa = input("Podaj nazwę pliku do zapisu: ")

        if 0 <= wsp_x1 < wsp_x2 <= d.shape[1] and 0 <= wsp_y1 < wsp_y2 <= d.shape[0]:
            sub_image = im.crop((wsp_x1, wsp_y1, wsp_x2, wsp_y2))
            sub_image.save(nazwa + ".png")
            print(f"Podobraz zapisano jako {nazwa}.png")
        else:
            print("Błąd: Niepoprawne współrzędne!")

def zadanie_6():
    import cv2
    print("\nTyp przekształcenia:")
    print("a - Mnożenie przez stałą")
    print("b - Transformacja logarytmiczna")
    print("c - Zmiana dynamiki skali szarości (rozciąganie kontrastu)")
    print("d - Korekcja gamma")

    wybor = input("Wybierz typ przekształcenia (a/b/c/d): ").lower()
    if wybor == 'a':
        print("Obrazy: chest-xray.tif, pollen-dark.tif, spectrum.tif")
        nazwa_pliku = input("Podaj nazwę pliku (np. chest-xray.tif): ")
    elif wybor == 'b':
        nazwa_pliku = "spectrum.tif"
    elif wybor == 'c':
        print("Obrazy: chest-xray.tif, einstein-low-contrast.tif, pollen-lowcontrast.tif")
        nazwa_pliku = input("Podaj nazwę pliku (np. chest-xray.tif): ")
    else:
        nazwa_pliku = "aerial_view.tif"
    try:
        img = Image.open(nazwa_pliku).convert("L")  # skalowanie do szarości
        img_np = np.array(img).astype(np.float32) / 255.0  # normalizacja
    except FileNotFoundError:
        print("Błąd: Plik nie został znaleziony.")
        return

    if wybor == 'a':
        c = float(input("Podaj wartość c: "))
        transformed = np.clip(c * img_np, 0, 1)

    elif wybor == 'b':
        c = float(input("Podaj wartość c (np. 1–100): "))

        img_np_log = img_np + 1
        transformed = c * np.log(img_np_log)
        transformed = np.clip(transformed, 0, 1)

    elif wybor == 'c':
        m = float(input("Podaj m (np. 0.45): "))
        e = float(input("Podaj e (np. 8): "))
        safe_r = np.maximum(img_np, 1e-2)
        transformed = 1 / (1 + (m / safe_r) ** e)


    elif wybor == 'd':
        c = float(input("Podaj wartość c (np. 1): "))
        gamma = float(input("Podaj wartość gamma (np. 4): "))
        transformed = c * (img_np ** gamma)
        transformed = np.clip(transformed, 0, 1)

    else:
        print("Nieprawidłowy wybór.")
        return

    # Wyświetlenie wyników
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_np, cmap='gray')
    plt.title("Oryginalny obraz")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(transformed, cmap='gray')
    plt.title("Po przekształceniu")
    plt.axis('off')
    if wybor == 'a':
        plt.suptitle("Przekształcenie a - Mnożenie przez stałą o wartości C = " + str(c) + " pliku " + nazwa_pliku, fontsize=16, y=0.975)
    elif wybor == 'b':
        plt.suptitle("Przekształcenie b - Transformacja logarytmiczna o wartości C = " + str(c) + " pliku " + nazwa_pliku, fontsize=15, y=0.99)
    elif wybor == 'c':
        plt.suptitle("Przekształcenie c - Zmiana dynamiki skali szarości (rozciąganie kontrastu) o wartości M = " + str(m) + " E = " + str(e) + "\n pliku " + nazwa_pliku, fontsize=15, y=0.99)
    elif wybor == 'd':
        plt.suptitle("Przekształcenie d - Korekcja gamma o wartości C = " + str(c) + " gamma = " + str(gamma) + " pliku " + nazwa_pliku, fontsize=15, y=0.99)
    plt.show()

def zadanie_7():
    import cv2

    print("\nZadanie 7: Wyrównywanie histogramu.")
    print("Obrazy: chest-xray.tif, pollen-dark.tif, pollen-ligt.tif,")
    print("        pollen-lowcontrast.tif, pout.tif, spectrum.tif")

    nazwa_pliku = input("Podaj nazwę pliku do wyrównania histogramu: ")
    try:
        img = Image.open(nazwa_pliku).convert('L')  # konwersja do skali szarości
        img_np = np.array(img)
    except FileNotFoundError:
        print("Błąd: Plik nie został znaleziony.")
        return

    # Wyrównywanie histogramu
    img_eq = cv2.equalizeHist(img_np)

    # Wyświetlenie obrazów przed i po
    plt.figure(figsize=(12, 7))

    plt.subplot(2, 2, 1)
    plt.imshow(img_np, cmap='gray')
    plt.title("Oryginalny obraz")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(img_eq, cmap='gray')
    plt.title("Po wyrównaniu histogramu")
    plt.axis('off')

    # Histogramy
    plt.subplot(2, 2, 3)
    plt.hist(img_np.ravel(), bins=256, range=(0, 256), color='gray')
    plt.title("Histogram - przed")
    plt.xlabel("Poziom jasności")
    plt.ylabel("Liczba pikseli")

    plt.subplot(2, 2, 4)
    plt.hist(img_eq.ravel(), bins=256, range=(0, 256), color='gray')
    plt.title("Histogram - po")
    plt.xlabel("Poziom jasności")
    plt.ylabel("Liczba pikseli")

    plt.tight_layout()
    plt.suptitle("Wyniki wyrownania histogramu dla obrazu " + nazwa_pliku, fontsize=14, y=1.0)
    plt.show()

def zadanie_8():
    print("\n===== ZADANIE 8 =====")
    print("a - Lokalne wyrównywanie histogramu")
    print("b - Poprawa jakości oparta na lokalnych statystykach")
    wybor = input("Wybierz metodę (a/b): ").lower()

    # Wczytanie obrazu
    nazwa_pliku = "hidden-symbols.tif"
    try:
        img = Image.open(nazwa_pliku).convert('L')  # skala szarości
        img_np = np.array(img)
    except FileNotFoundError:
        print("Błąd: Nie znaleziono pliku.")
        return
    if wybor == 'a':
        # Normalizacja do zakresu [0, 255] i przygotowanie obrazu wyjściowego
        img_np = img_np.astype(np.uint8)
        out_img = np.zeros_like(img_np)

        # Padujemy obraz by móc działać na krawędziach
        padded_img = np.pad(img_np, pad_width=1, mode='reflect')

        # Przechodzimy po każdym pikselu (bez ramek, bo są w paddingu)
        for i in range(img_np.shape[0]):
            for j in range(img_np.shape[1]):
                window = padded_img[i:i + 3, j:j + 3]

                hist, _ = np.histogram(window, bins=256, range=(0, 256))

                cdf = hist.cumsum()
                cdf = cdf * 255 / cdf[-1]  # normalizacja do [0,255]

                pixel_value = padded_img[i + 1, j + 1]
                new_value = cdf[pixel_value]
                out_img[i, j] = new_value

        # Wyświetlanie
        plt.figure(figsize=(11, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(img_np, cmap='gray')
        plt.title("Oryginalny obraz")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        global_eq = exposure.equalize_hist(img_np)
        plt.imshow(global_eq, cmap='gray')
        plt.title("Globalne wyrównanie histogramu")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(out_img, cmap='gray')
        plt.title("Lokalne wyrównanie (3x3)")
        plt.axis('off')
        plt.tight_layout()
        plt.suptitle("a - Lokalne wyrównywanie histogramu pliku " + nazwa_pliku, fontsize=16, y=0.975)
        plt.show()


    elif wybor == 'b':
        # Normalizacja obrazu do zakresu [0, 1]
        img_norm = img_np.astype(np.float32) / 255.0

        # Parametry
        C = 22.8
        k0 = k2 = 0.0
        k1 = k3 = 0.1

        from scipy.ndimage import uniform_filter

        # Średnia i odchylenie dla całego obrazu
        mG = np.mean(img_norm)
        sigmaG = np.std(img_norm)

        # Średnia i wariancja w oknie 3x3
        mSxy = uniform_filter(img_norm, size=3)
        sigmaSxy = np.sqrt(uniform_filter(img_norm ** 2, size=3) - mSxy ** 2)

        # Tworzenie maski spełniającej warunki
        mask = (
                (mSxy >= k0 * mG) & (mSxy <= k1 * mG) &
                (sigmaSxy >= k2 * sigmaG) & (sigmaSxy <= k3 * sigmaG)
        )

        # Modyfikacja obrazu według wzoru
        enhanced = img_norm.copy()
        enhanced[mask] = C * img_norm[mask]
        enhanced = np.clip(enhanced, 0, 1)

        # Wyświetlenie obrazów
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img_norm, cmap='gray')
        plt.title("Oryginalny obraz")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(enhanced, cmap='gray')
        plt.title("Po lokalnej poprawie jakości")
        plt.axis('off')
        plt.tight_layout()
        plt.suptitle("b - Poprawa jakości oparta na lokalnych statystykach pliku " + nazwa_pliku, fontsize=16, y=0.075)
        plt.show()

    else:
        print("Nieprawidłowy wybór.")

def zadanie_9():
    print("\n===== ZADANIE 9: Filtracja szumu 'sól i pieprz' =====")
    print("Obrazy: cboard_pepper_only.tif, cboard_salt_only.tif, cboard_salt_pepper.tif")
    nazwa_pliku = input("Podaj nazwę pliku (np. cboard_salt_pepper.tif): ")

    try:
        img = Image.open(nazwa_pliku).convert('L')
        img_np = np.array(img)
    except FileNotFoundError:
        print("Błąd: Plik nie został znaleziony.")
        return

    print("a - Filtr średniej (splot z maską)")
    print("b - Filtr medianowy")
    print("c - Filtr minimum i maksimum")

    wybor = input("Wybierz typ filtra (a/b/c): ").lower()

    size = int(input("Podaj rozmiar maski (np. 3): "))
    if size % 2 == 0:
        print("Rozmiar maski musi być nieparzysty!")
        return

    if wybor == 'a':
        # Maska średniej
        mask = np.ones((size, size)) / (size * size)
        wynik = convolve2d(img_np, mask, mode='same', boundary='symm')
        title = "Filtr średniej (splot)"

        plt.figure(figsize=(10, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(img_np, cmap='gray')
        plt.title("Oryginalny obraz")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(wynik, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.suptitle("A - Filtr średniej (splot z maską) dla maski o rozmiarze " + str(size) + "x" + str(size) + " pliku " + nazwa_pliku, fontsize=16, y=0.95)
        plt.show()

    elif wybor == 'b':
        wynik = median_filter(img_np, size=size)
        title = "Filtr medianowy"

        plt.figure(figsize=(10, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(img_np, cmap='gray')
        plt.title("Oryginalny obraz")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(wynik, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.suptitle("B - Filtr medianowy dla maski o rozmiarze " + str(size) + "x" + str(size) + " pliku " + nazwa_pliku, fontsize=16, y=0.95)
        plt.show()

    elif wybor == 'c':
        wynik_min = minimum_filter(img_np, size=size)
        wynik_max = maximum_filter(img_np, size=size)

        plt.figure(figsize=(15, 7))
        plt.subplot(1, 3, 1)
        plt.imshow(img_np, cmap='gray')
        plt.title("Oryginalny obraz")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(wynik_min, cmap='gray')
        plt.title("Filtr minimum (usuwa sól)")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(wynik_max, cmap='gray')
        plt.title("Filtr maksimum (usuwa pieprz)")
        plt.axis('off')

        plt.tight_layout()
        plt.suptitle("C - Filtr minimum i maksimum dla maski o rozmiarze " + str(size) + "x" + str(size) + " pliku " + nazwa_pliku, fontsize=16, y=0.95)
        plt.show()

    else:
        print("Nieprawidłowy wybór.")

def zadanie_10():
    print("\n===== ZADANIE 10: Filtry dolnoprzepustowe =====")
    print("Obrazy: characters_test_pattern.tif, zoneplate.tif")
    nazwa_pliku = input("Podaj nazwę pliku (np. characters_test_pattern.tif): ")

    try:
        img = Image.open(nazwa_pliku).convert('L')
        img_np = np.array(img)
    except FileNotFoundError:
        print("Błąd: Nie znaleziono pliku.")
        return

    print("a - Filtr uśredniający")
    print("b - Filtr Gaussa")

    wybor = input("Wybierz typ filtra (a/b): ").lower()

    if wybor == 'a':
        size = int(input("Podaj rozmiar maski (np. 3, 5, 9): "))
        wynik = uniform_filter(img_np, size=size)
        title = f"Filtr uśredniający ({size}×{size})"

    elif wybor == 'b':
        sigma = float(input("Podaj wartość sigma (np. 1.0, 2.0): "))
        wynik = gaussian_filter(img_np, sigma=sigma)
        title = f"Filtr Gaussa (σ = {sigma})"

    else:
        print("Nieprawidłowy wybór.")
        return

    # Wyświetlenie
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_np, cmap='gray')
    plt.title("Oryginalny obraz")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(wynik, cmap='gray')
    plt.title(title)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def zadanie_11():
    print("\n===== ZADANIE 11: Filtracja górnoprzepustowa =====")
    print("a - Filtr Sobela (krawędzie)")
    print("b - Laplasjan (wyostrzanie)")
    print("c - Unsharp masking / High-boost filtering")
    wybor = input("Wybierz metodę (a/b/c): ").lower()

    # Wczytanie obrazu
    if wybor == "a":
        nazwa_pliku = "circuitmask.tif"
    elif wybor == "b":
        nazwa_pliku = "blurry-moon.tif"
    elif wybor == "c":
        nazwa_pliku = "text-dipxe-blurred.tif"
    try:
        img = Image.open(nazwa_pliku).convert('L')
        img_np = np.array(img).astype(np.float32)
    except FileNotFoundError:
        print("Błąd: Plik nie został znaleziony.")
        return

    if wybor == 'a':
        # Filtry Sobela – poziomy i pionowy
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]])

        gx = convolve(img_np, sobel_x)
        gy = convolve(img_np, sobel_y)
        magnitude = np.hypot(gx, gy)
        magnitude = magnitude / np.max(magnitude) * 255

        plt.figure(figsize=(10, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(img_np, cmap='gray')
        plt.title("Oryginalny obraz")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(magnitude, cmap='gray')
        plt.title("Filtr Sobela – wykryte krawędzie")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    elif wybor == 'b':
        # Laplasjan
        laplacian_mask = np.array([[0, -1, 0],
                                   [-1, 4, -1],
                                   [0, -1, 0]])
        laplacian = convolve(img_np, laplacian_mask)
        sharpened = img_np + laplacian
        sharpened = np.clip(sharpened, 0, 255)

        plt.figure(figsize=(15, 7))
        plt.subplot(1, 3, 1)
        plt.imshow(img_np, cmap='gray')
        plt.title("Oryginalny obraz")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(laplacian, cmap='gray')
        plt.title("Laplasjan")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(sharpened, cmap='gray')
        plt.title("Po filtrze Laplasjana")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    elif wybor == 'c':
        print("Tryb: Unsharp masking lub High-boost filtering")
        A = float(input("Podaj wartość A (=1 dla unsharp, >1 dla high-boost): "))
        sigma = float(input("Podaj wartość sigma dla rozmycia Gaussa (np. 5.0): "))

        blur = gaussian_filter(img_np, sigma=sigma)
        mask = img_np - blur
        highboost = img_np + A * mask
        highboost = np.clip(highboost, 0, 255)

        plt.figure(figsize=(10, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(img_np, cmap='gray')
        plt.title("Oryginalny obraz")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(highboost, cmap='gray')
        if A == 1.0:
            plt.title("Unsharp masking")
        else:
            plt.title(f"High-boost filtering (A={A})")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    else:
        print("Nieprawidłowy wybór.")

def menu():
    while True:
        print("\n===== MENU GŁÓWNE =====")
        print("Wybierz numer zadania (5–12):")
        print("5 - Zadanie: wczytanie, profil jasności, podobrazy")
        print("6 - Zadanie: przekształcenia")
        print("7 - Zadanie: wyrównywanie histogramu")
        print("8 - Zadanie: lokalne wyrównywanie kontrastu")
        print("9 - Zadanie: filtracja dolnoprzepustowa (sól i pieprz)")
        print("10 - Zadanie: filtry dolnoprzepustowe (uśredniający i Gaussa)")
        print("11 - Zadanie: filtracja górnoprzepustowa (krawędzie, wyostrzanie)")
        print("0 - Wyjście z programu")
        wybor = input("Twój wybór: ")

        if wybor == "5":
            zadanie_5()
        elif wybor == "6":
            zadanie_6()
        elif wybor == "7":
            zadanie_7()
        elif wybor == "8":
            zadanie_8()
        elif wybor == "9":
            zadanie_9()
        elif wybor == "10":
            zadanie_10()
        elif wybor == "11":
            zadanie_11()
        elif wybor == "0":
            print("Zamykanie programu...")
            break
        else:
            print("Nieprawidłowy wybór. Wybierz coś z listy.")

# Start programu
menu()
